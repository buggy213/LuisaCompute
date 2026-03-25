#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/passes/coro_normalize_cf.h>

namespace luisa::compute::xir {

namespace detail {

// TODO: implement SimpleLoop → Loop conversion and Break/Continue elimination.
// For now, this pass validates that the function is already normalized
// (only IfInst, SwitchInst, LoopInst as CF) and runs early return elimination.

static void coro_normalize_cf_pass(FunctionDefinition *function) noexcept {
    LUISA_ASSERT(function != nullptr, "Null function passed to coro_normalize_cf_pass.");
    LUISA_ASSERT(function->is_coroutine(), "Function is not marked as a coroutine.");

    // Run early return elimination first
    static_cast<void>(early_return_elimination_pass_run_on_function(function));

    // Validate: no SimpleLoopInst, BreakInst, or ContinueInst should remain
    function->traverse_instructions([](Instruction *inst) noexcept {
        LUISA_ASSERT(!inst->isa<SimpleLoopInst>(),
                     "SimpleLoopInst found after CF normalization. "
                     "Please convert SimpleLoopInst to LoopInst before running coroutine passes.");
        LUISA_ASSERT(!inst->isa<BreakInst>(),
                     "BreakInst found after CF normalization. "
                     "Please eliminate BreakInst before running coroutine passes.");
        LUISA_ASSERT(!inst->isa<ContinueInst>(),
                     "ContinueInst found after CF normalization. "
                     "Please eliminate ContinueInst before running coroutine passes.");
    });

    // Validate: only one ReturnInst at the very end
    ReturnInst *final_return = nullptr;
    function->traverse_instructions([&](Instruction *inst) noexcept {
        if (inst->isa<ReturnInst>()) {
            LUISA_ASSERT(final_return == nullptr,
                         "Multiple ReturnInst found after early return elimination.");
            final_return = static_cast<ReturnInst *>(inst);
        }
    });
}

}// namespace detail

void coro_normalize_cf_pass_run_on_function(FunctionDefinition *function) noexcept {
    detail::coro_normalize_cf_pass(function);
}

void coro_normalize_cf_pass_run_on_module(Module *module) noexcept {
    for (auto f : module->function_list()) {
        if (auto f_defn = f->definition(); f_defn != nullptr && f_defn->is_coroutine()) {
            detail::coro_normalize_cf_pass(f_defn);
        }
    }
}


}// namespace luisa::compute::xir
