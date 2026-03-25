#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class FunctionDefinition;

// Control flow normalization for coroutine functions.
// After this pass, only IfInst, SwitchInst, and LoopInst remain as
// control flow. SimpleLoopInst, BreakInst, ContinueInst, and early
// ReturnInst are all eliminated.
//
// Must be run before CoroGraph analysis.
LUISA_XIR_API void coro_normalize_cf_pass(FunctionDefinition *function) noexcept;

}// namespace luisa::compute::xir
