//! # Stackful Coroutine and Context Switch (riscv64)
//!
//! In this exercise, you implement the minimal context switch using inline assembly,
//! which is the core mechanism of OS thread scheduling. This crate is **riscv64 only**;
//! run `cargo test` on riscv64 Linux, or use the repo's normal flow (`./check.sh` / `oscamp`) on x86 with QEMU.
//!
//! ## Key Concepts
//! - **Callee-saved registers**: Save and restore them on switch so the switched-away task can resume correctly later.
//! - **Stack pointer `sp`** and **return address `ra`**: Restore them in the new context; the first time we switch to a task, `ret` jumps to `ra` (the entry point).
//! - Naked assembly: `core::arch::naked_asm!`
//!
//! ## riscv64 ABI (for this exercise)
//! - Callee-saved: `sp`, `ra`, `s0`–`s11`. The `ret` instruction is `jalr zero, 0(ra)`.
//! - First and second arguments: `a0` (old context), `a1` (new context).

#![cfg(target_arch = "riscv64")]

/// Saved integer register state for one task (riscv64). Layout must match the asm offsets:
/// `sp` at 0, `ra` at 8, then `s0`–`s11` at 16, 24, … 104.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TaskContext {
    pub sp: u64,
    pub ra: u64,
    pub s0: u64,
    pub s1: u64,
    pub s2: u64,
    pub s3: u64,
    pub s4: u64,
    pub s5: u64,
    pub s6: u64,
    pub s7: u64,
    pub s8: u64,
    pub s9: u64,
    pub s10: u64,
    pub s11: u64,
}

impl TaskContext {
    pub const fn empty() -> Self {
        Self {
            sp: 0,
            ra: 0,
            s0: 0,
            s1: 0,
            s2: 0,
            s3: 0,
            s4: 0,
            s5: 0,
            s6: 0,
            s7: 0,
            s8: 0,
            s9: 0,
            s10: 0,
            s11: 0,
        }
    }

    /// Initialize this context so that when we switch to it, execution starts at `entry`.
    ///
    /// - Set `ra = entry` so that the first `ret` in the new context jumps to `entry`.
    /// - Set `sp = stack_top` with 16-byte alignment (RISC-V ABI requires 16-byte aligned stack at function entry).
    /// - Leave `s0`–`s11` zero; they will be loaded on switch.
    pub fn init(&mut self, stack_top: usize, entry: usize) {
        *self = Self::empty();
        self.ra = entry as u64;
        self.sp = (stack_top & !15) as u64;
    }
}

/// Switch from `old` to `new`, preserving the RISC-V callee-saved registers.
///
/// The integer context keeps its fixed layout. Floating-point callee-saved registers,
/// `fcsr`, and the caller's return address live in a 16-byte-aligned frame on the
/// suspended stack; `old.ra` resumes at the assembly epilogue that restores them.
/// A newly initialized context instead jumps directly to its entry.
///
/// # Safety
/// `new` must be initialized with a live, sufficiently large stack or have been saved
/// by this function. Contexts must not overlap or be resumed concurrently. An initial
/// entry must switch away permanently on completion; it must not return or unwind.
/// Both stacks and contexts must remain live until their suspended execution ends.
pub unsafe fn switch_context(old: &mut TaskContext, new: &TaskContext) {
    switch_context_raw(old, new);
}

// Keep the public Rust ABI while fixing the assembly argument registers to a0/a1.
#[unsafe(naked)]
unsafe extern "C" fn switch_context_raw(_old: *mut TaskContext, _new: *const TaskContext) {
    core::arch::naked_asm!(
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

const STACK_SIZE: usize = 1024 * 64;

/// Allocate a stack for a coroutine. Returns `(buffer, stack_top)` where `stack_top` is the high address
/// (stack grows down). The buffer must be kept alive for the lifetime of the context using this stack.
pub fn alloc_stack() -> (Vec<u8>, usize) {
    let buffer = vec![0; STACK_SIZE];
    let stack_top = (buffer.as_ptr() as usize + buffer.len()) & !15;
    (buffer, stack_top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_entry() {
        COUNTER.store(42, Ordering::SeqCst);
        loop {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn test_alloc_stack() {
        let (buf, top) = alloc_stack();
        assert_eq!(top, buf.as_ptr() as usize + STACK_SIZE);
        assert!(top % 16 == 0);
    }

    #[test]
    fn test_context_init() {
        let (buf, top) = alloc_stack();
        let _ = buf;
        let mut ctx = TaskContext::empty();
        let entry = task_entry as *const () as usize;
        ctx.init(top, entry);
        assert_eq!(ctx.ra, entry as u64);
        assert!(ctx.sp != 0);
    }

    #[test]
    fn test_switch_to_task() {
        COUNTER.store(0, Ordering::SeqCst);

        static mut MAIN_CTX_PTR: *mut TaskContext = std::ptr::null_mut();
        static mut TASK_CTX_PTR: *mut TaskContext = std::ptr::null_mut();

        extern "C" fn cooperative_task() {
            COUNTER.store(99, Ordering::SeqCst);
            unsafe {
                switch_context(&mut *TASK_CTX_PTR, &*MAIN_CTX_PTR);
            }
        }

        let (_stack_buf, stack_top) = alloc_stack();
        let mut main_ctx = TaskContext::empty();
        let mut task_ctx = TaskContext::empty();
        task_ctx.init(stack_top, cooperative_task as *const () as usize);

        unsafe {
            MAIN_CTX_PTR = &mut main_ctx;
            TASK_CTX_PTR = &mut task_ctx;
            switch_context(&mut main_ctx, &task_ctx);
        }

        assert_eq!(COUNTER.load(Ordering::SeqCst), 99);
    }
}
