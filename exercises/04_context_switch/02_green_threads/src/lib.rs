//! # Green Thread Scheduler (riscv64)
//!
//! In this exercise, you build a simple cooperative (green) thread scheduler on top of context switching.
//! This crate is **riscv64 only**; run with the repo's normal flow (`./check.sh` / `oscamp`) or natively on riscv64.
//!
//! ## Key Concepts
//! - Cooperative vs preemptive scheduling
//! - Thread state: `Ready`, `Running`, `Finished`
//! - `yield_now()`: current thread voluntarily gives up the CPU
//! - Scheduler loop: pick next ready thread and switch to it
//!
//! ## Design
//! Each green thread has its own stack and `TaskContext`. Threads call `yield_now()` to yield.
//! The scheduler round-robins among ready threads. User entry is wrapped by `thread_wrapper`, which
//! calls the entry then marks the thread `Finished` and switches back.

#![cfg(target_arch = "riscv64")]

use core::arch::naked_asm;
use std::cell::Cell;

/// Per-thread stack size. Slightly larger to avoid overflow under QEMU / test harness.
const STACK_SIZE: usize = 1024 * 128;

/// Task context (riscv64); layout must match `01_stack_coroutine::TaskContext` and the asm below.
#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct TaskContext {
    sp: u64,
    ra: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    s8: u64,
    s9: u64,
    s10: u64,
    s11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadState {
    Ready,
    Running,
    Finished,
}

struct GreenThread {
    ctx: TaskContext,
    state: ThreadState,
    _stack: Option<Vec<u8>>,
    /// User entry; taken once when the thread is first scheduled and passed to `thread_wrapper`.
    entry: Option<extern "C" fn()>,
}

// Active scheduler and initial entry belong to this OS thread. Separate native
// threads may run independent schedulers without sharing register contexts.
std::thread_local! {
    static CURRENT_THREAD_ENTRY: Cell<Option<extern "C" fn()>> = const { Cell::new(None) };
    static SCHEDULER: Cell<*mut Scheduler> = const { Cell::new(std::ptr::null_mut()) };
}

/// Wrapper run as the initial `ra` for each green thread: call the user entry (from `CURRENT_THREAD_ENTRY`), then mark Finished and switch back.
extern "C" fn thread_wrapper() {
    let entry = CURRENT_THREAD_ENTRY.with(Cell::take).expect("missing green-thread entry");
    entry();
    thread_finished();
}

/// Save integer callee-saved registers in the context and floating-point state in
/// a suspended stack frame. Saved `ra` resumes the frame-restoring epilogue;
/// a fresh context enters `thread_wrapper` directly.
/// Zero `a0`/`a1` before switching so context pointers do not leak to a fresh entry.
///
/// Must be `#[unsafe(naked)]` to prevent the compiler from generating a prologue/epilogue.
#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: *mut TaskContext, _new: *const TaskContext) {
    naked_asm!(
        ".option push",
        ".option arch, +d",
        "addi sp, sp, -112",
        "sd ra, 0(sp)",
        "frcsr t0",
        "sd t0, 8(sp)",
        "fsd fs0, 16(sp)",
        "fsd fs1, 24(sp)",
        "fsd fs2, 32(sp)",
        "fsd fs3, 40(sp)",
        "fsd fs4, 48(sp)",
        "fsd fs5, 56(sp)",
        "fsd fs6, 64(sp)",
        "fsd fs7, 72(sp)",
        "fsd fs8, 80(sp)",
        "fsd fs9, 88(sp)",
        "fsd fs10, 96(sp)",
        "fsd fs11, 104(sp)",
        "sd sp, 0(a0)",
        "lla t0, 2f",
        "sd t0, 8(a0)",
        "sd s0, 16(a0)",
        "sd s1, 24(a0)",
        "sd s2, 32(a0)",
        "sd s3, 40(a0)",
        "sd s4, 48(a0)",
        "sd s5, 56(a0)",
        "sd s6, 64(a0)",
        "sd s7, 72(a0)",
        "sd s8, 80(a0)",
        "sd s9, 88(a0)",
        "sd s10, 96(a0)",
        "sd s11, 104(a0)",
        "ld sp, 0(a1)",
        "ld ra, 8(a1)",
        "ld s0, 16(a1)",
        "ld s1, 24(a1)",
        "ld s2, 32(a1)",
        "ld s3, 40(a1)",
        "ld s4, 48(a1)",
        "ld s5, 56(a1)",
        "ld s6, 64(a1)",
        "ld s7, 72(a1)",
        "ld s8, 80(a1)",
        "ld s9, 88(a1)",
        "ld s10, 96(a1)",
        "ld s11, 104(a1)",
        "li a0, 0",
        "li a1, 0",
        "ret",
        "2:",
        "fld fs0, 16(sp)",
        "fld fs1, 24(sp)",
        "fld fs2, 32(sp)",
        "fld fs3, 40(sp)",
        "fld fs4, 48(sp)",
        "fld fs5, 56(sp)",
        "fld fs6, 64(sp)",
        "fld fs7, 72(sp)",
        "fld fs8, 80(sp)",
        "fld fs9, 88(sp)",
        "fld fs10, 96(sp)",
        "fld fs11, 104(sp)",
        "ld t0, 8(sp)",
        "fscsr t0",
        "ld ra, 0(sp)",
        "addi sp, sp, 112",
        "ret",
        ".option pop",
    );
}

pub struct Scheduler {
    threads: Vec<GreenThread>,
    current: usize,
}

impl Scheduler {
    pub fn new() -> Self {
        let main_thread = GreenThread {
            ctx: TaskContext::default(),
            state: ThreadState::Running,
            _stack: None,
            entry: None,
        };

        Self {
            threads: vec![main_thread],
            current: 0,
        }
    }

    /// Register a new green thread that will run `entry` when first scheduled.
    ///
    /// 1. Allocate a stack of `STACK_SIZE` bytes; compute `stack_top` (high address).
    /// 2. Set up the context: `ra = thread_wrapper` so the first switch jumps to the wrapper;
    ///    `sp` must be 16-byte aligned (e.g. `(stack_top - 16) & !15` to leave headroom).
    /// 3. Push a `GreenThread` with this context, state `Ready`, and `entry` stored for the wrapper to call.
    pub fn spawn(&mut self, entry: extern "C" fn()) {
        let stack = vec![0; STACK_SIZE];
        let stack_top = stack.as_ptr() as usize + stack.len();
        let ctx = TaskContext {
            sp: (stack_top & !15) as u64,
            ra: thread_wrapper as *const () as u64,
            ..TaskContext::default()
        };
        self.threads.push(GreenThread {
            ctx,
            state: ThreadState::Ready,
            _stack: Some(stack),
            entry: Some(entry),
        });
    }

    /// Run the scheduler until all threads (except the main one) are `Finished`.
    ///
    /// The main context participates in round-robin scheduling but is never
    /// finished. Keep all stacks owned until execution returns to the main stack.
    /// Thread-local scheduler state is restored afterward, including for nested runs.
    pub fn run(&mut self) {
        let scheduler = self as *mut Self;
        let previous = SCHEDULER.with(|active| active.replace(scheduler));
        // Do not keep a Rust reference to the scheduler alive across a switch:
        // the resumed thread must be free to update the same scheduling state.
        unsafe {
            while (&(*scheduler).threads)[1..]
                .iter()
                .any(|thread| thread.state != ThreadState::Finished)
            {
                Self::schedule_next(scheduler);
            }
        }
        SCHEDULER.with(|active| active.set(previous));
    }

    /// Find the next ready thread (starting from `current + 1` round-robin), mark current as `Ready` (if not `Finished`), mark next as `Running`, set `CURRENT_THREAD_ENTRY` if the next thread has an entry, then switch to it.
    unsafe fn schedule_next(scheduler: *mut Self) {
        let (old_context, new_context) = {
            let scheduler = &mut *scheduler;
            let current = scheduler.current;
            let count = scheduler.threads.len();
            let next = (1..count)
                .map(|offset| (current + offset) % count)
                .find(|&index| scheduler.threads[index].state == ThreadState::Ready);
            let Some(next) = next else {
                return;
            };

            if scheduler.threads[current].state != ThreadState::Finished {
                scheduler.threads[current].state = ThreadState::Ready;
            }
            scheduler.threads[next].state = ThreadState::Running;
            scheduler.current = next;
            let entry = scheduler.threads[next].entry.take();
            CURRENT_THREAD_ENTRY.with(|pending| pending.set(entry));

            let old_context = scheduler.threads[current].ctx.as_mut_ptr();
            let new_context = scheduler.threads[next].ctx.as_ptr();
            (old_context, new_context)
        };
        // Only raw pointers cross the switch; the vector cannot move while run()
        // holds the scheduler, and each thread's stack remains allocated.
        switch_context(old_context, new_context);
    }
}

impl TaskContext {
    fn as_mut_ptr(&mut self) -> *mut TaskContext {
        self as *mut TaskContext
    }
    fn as_ptr(&self) -> *const TaskContext {
        self as *const TaskContext
    }
}

/// Current thread voluntarily yields; the scheduler will pick the next ready thread.
pub fn yield_now() {
    let scheduler = SCHEDULER.with(Cell::get);
    if !scheduler.is_null() {
        unsafe { Scheduler::schedule_next(scheduler) };
    }
}

/// Mark current thread as `Finished` and switch to the next (called by `thread_wrapper` after the user entry returns).
fn thread_finished() -> ! {
    let scheduler = SCHEDULER.with(Cell::get);
    assert!(!scheduler.is_null(), "green thread completed outside its scheduler");
    unsafe {
        {
            let scheduler = &mut *scheduler;
            scheduler.threads[scheduler.current].state = ThreadState::Finished;
        }
        Scheduler::schedule_next(scheduler);
    }
    unreachable!("a finished green thread must never be resumed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Tests must run serially: the scheduler uses global state (SCHEDULER, CURRENT_THREAD_ENTRY).
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    static EXEC_ORDER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_a() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(100, Ordering::SeqCst);
    }

    extern "C" fn task_b() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
    }

    #[test]
    fn test_scheduler_runs_all() {
        let _guard = TEST_LOCK.lock().unwrap();
        EXEC_ORDER.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(task_a);
        sched.spawn(task_b);
        sched.run();

        let got = EXEC_ORDER.load(Ordering::SeqCst);
        if got != 122 {
            panic!(
                "EXEC_ORDER: expected 122, got {} (run with --nocapture to see stderr)",
                got
            );
        }
    }

    static SIMPLE_FLAG: AtomicU32 = AtomicU32::new(0);

    extern "C" fn simple_task() {
        SIMPLE_FLAG.store(42, Ordering::SeqCst);
    }

    #[test]
    fn test_single_thread() {
        let _guard = TEST_LOCK.lock().unwrap();
        SIMPLE_FLAG.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(simple_task);
        sched.run();

        assert_eq!(SIMPLE_FLAG.load(Ordering::SeqCst), 42);
    }
}
