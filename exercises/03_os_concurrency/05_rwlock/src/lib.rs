//! # Read-Write Lock (Writer-Priority)
//!
//! In this exercise, you will implement a **writer-priority** read-write lock from scratch using atomics.
//! Multiple readers may hold the lock concurrently; a writer holds it exclusively.
//!
//! **Note:** Rust's standard library already provides [`std::sync::RwLock`]. This exercise implements
//! a minimal version for learning the protocol and policy without using the standard one.
//!
//! ## Common policies for read-write locks
//! Different implementations can give different **priority** when both readers and writers are waiting:
//!
//! - **Reader-priority (读者优先)**: New readers are allowed to enter while a writer is waiting, so writers
//!   may be starved if readers keep arriving.
//! - **Writer-priority (写者优先)**: Once a writer is waiting, no new readers are admitted until that writer
//!   has run; this exercise implements this policy.
//! - **Read-write fair (读写公平)**: Requests are served in a fair order (e.g. FIFO or round-robin), so
//!   neither readers nor writers are systematically starved.
//!
//! ## Key Concepts
//! - **Readers**: share access; many threads can hold a read lock at once.
//! - **Writer**: exclusive access; only one writer, and no readers while the writer holds the lock.
//! - **Writer-priority (this implementation)**: when at least one writer is waiting, new readers block
//!   until the writer runs.
//!
//! ## State (single atomic)
//! One `AtomicU64` tracks the complete protocol: bits 0..30 count readers,
//! bit 30 marks an active writer, bit 31 marks waiting writers, and bits 32..64
//! count waiting writers. The waiting flag remains set across writer handoffs
//! until the last queued writer acquires the lock, preventing reader barging.
//! A count is necessary: a single waiting bit cannot represent multiple waiters
//! when the active writer releases. No `std::sync::RwLock` is used.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of concurrent readers (fits in state bits).
const READER_MASK: u64 = (1 << 30) - 1;
/// Bit set when a writer holds the lock.
const WRITER_HOLDING: u64 = 1 << 30;
/// Bit set when at least one writer is waiting (writer-priority: block new readers).
const WRITER_WAITING: u64 = 1 << 31;
/// One queued writer in the upper half of the state.
const WRITER_UNIT: u64 = 1 << 32;
const WRITER_COUNT_MASK: u64 = u64::MAX << 32;

/// Writer-priority read-write lock. Implemented from scratch; does not use `std::sync::RwLock`.
pub struct RwLock<T> {
    state: AtomicU64,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicU64::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire a read lock. Blocks (spins) until no writer holds and no writer is waiting (writer-priority).
    ///
    /// Reader admission and writer registration modify the same atomic state,
    /// so a newly registered writer prevents any later reader from entering.
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if state & (WRITER_HOLDING | WRITER_WAITING) != 0
                || state & READER_MASK == READER_MASK
            {
                core::hint::spin_loop();
                state = self.state.load(Ordering::Acquire);
                continue;
            }
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return RwLockReadGuard { lock: self },
                Err(actual) => state = actual,
            }
        }
    }

    /// Acquire the write lock. Blocks until no readers and no other writer.
    ///
    /// Register as waiting before competing for exclusive access. Registration,
    /// acquisition and release preserve other writers' registrations, so new
    /// readers remain blocked until all pending writers have acquired the lock.
    /// Writers are not FIFO; a sustained stream of writers can starve readers.
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            // Never let the queued-writer count wrap into another state.
            if state & WRITER_COUNT_MASK == WRITER_COUNT_MASK {
                core::hint::spin_loop();
                state = self.state.load(Ordering::Acquire);
                continue;
            }
            let registered = (state + WRITER_UNIT) | WRITER_WAITING;
            match self.state.compare_exchange_weak(
                state,
                registered,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    state = registered;
                    break;
                }
                Err(actual) => state = actual,
            }
        }

        loop {
            if state & (READER_MASK | WRITER_HOLDING) != 0 {
                core::hint::spin_loop();
                state = self.state.load(Ordering::Acquire);
                continue;
            }
            // Convert our queued registration into exclusive ownership.
            let mut acquired = (state - WRITER_UNIT) | WRITER_HOLDING;
            if acquired & WRITER_COUNT_MASK == 0 {
                acquired &= !WRITER_WAITING;
            }
            match self.state.compare_exchange_weak(
                state,
                acquired,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return RwLockWriteGuard { lock: self },
                Err(actual) => state = actual,
            }
        }
    }
}

/// Guard for a read lock; releases the read lock on drop.
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Every live read guard contributes a reader, excluding all writers.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// Guard for a write lock; releases the write lock on drop.
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // The live write guard owns exclusive access to the data.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // The mutable guard borrow also excludes references through this guard.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        // Preserve queued writers and their flag across the handoff.
        self.lock
            .state
            .fetch_and(!WRITER_HOLDING, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_multiple_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(*g, 0);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_writer_excludes_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let lock_w = Arc::clone(&lock);
        let writer = thread::spawn(move || {
            let mut g = lock_w.write();
            *g = 42;
        });
        writer.join().unwrap();
        let g = lock.read();
        assert_eq!(*g, 42);
    }

    #[test]
    fn test_concurrent_reads_after_write() {
        let lock = Arc::new(RwLock::new(Vec::<i32>::new()));
        {
            let mut g = lock.write();
            g.push(1);
            g.push(2);
        }
        let mut handles = vec![];
        for _ in 0..5 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(g.len(), 2);
                assert_eq!(&*g, &[1, 2]);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_writes_serialized() {
        let lock = Arc::new(RwLock::new(0u64));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut g = l.write();
                    *g += 1;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*lock.read(), 1000);
    }
}
