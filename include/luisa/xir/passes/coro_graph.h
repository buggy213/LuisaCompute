#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Instruction;
class Value;
class FunctionDefinition;

using CoroInstrRef = uint32_t;
using CoroScopeRef = uint32_t;
static constexpr CoroInstrRef coro_invalid_ref = ~0u;

// An entry in the condition stack replay. Records the condition value
// that was active on the path leading to a suspend point.
struct ConditionStackItem {
    Instruction *condition;// The If/Switch condition instruction
    int32_t value;         // 0/1 for If, case value for Switch
};

// A switch case in the CoroGraph.
struct CoroSwitchCase {
    int32_t value;
    luisa::vector<CoroInstrRef> body;
};

// A node in the CoroGraph. Wraps IR instructions into a simplified
// control flow graph suitable for scope splitting.
//
// Tag meanings:
// - SIMPLE: a non-CF IR instruction
// - CONDITION_STACK_REPLAY: replays condition values to avoid framing them
// - MAKE_FIRST_FLAG: creates a first-execution boolean flag (= true)
// - SKIP_IF_FIRST_FLAG: conditionally skips body on first execution
// - CLEAR_FIRST_FLAG: clears the first flag to false
// - LOOP: a structured loop (prepare/body/update from LoopInst)
// - IF: conditional branch
// - SWITCH: multi-way branch
// - SUSPEND: coroutine yield point with a token
// - TERMINATE: function return / end of control flow
struct CoroInstruction {

    enum Tag {
        SIMPLE,
        CONDITION_STACK_REPLAY,
        MAKE_FIRST_FLAG,
        SKIP_IF_FIRST_FLAG,
        CLEAR_FIRST_FLAG,
        LOOP,
        IF,
        SWITCH,
        SUSPEND,
        TERMINATE,
    };

    Tag tag;

    // SIMPLE: the original IR instruction
    Instruction *ir_inst{nullptr};

    // SKIP_IF_FIRST_FLAG / CLEAR_FIRST_FLAG: the first flag ref
    // IF: condition instruction ref
    // SWITCH: value instruction ref
    CoroInstrRef ref{coro_invalid_ref};

    // SUSPEND: token
    uint32_t token{~0u};

    // SKIP_IF_FIRST_FLAG: body
    // LOOP: body block instructions
    // IF: true branch
    luisa::vector<CoroInstrRef> body;

    // IF: false branch
    // SWITCH: default branch
    luisa::vector<CoroInstrRef> else_body;

    // LOOP: prepare block instructions (condition computation)
    luisa::vector<CoroInstrRef> prepare;

    // LOOP: update block instructions
    luisa::vector<CoroInstrRef> update;

    // SWITCH: cases
    luisa::vector<CoroSwitchCase> cases;

    // CONDITION_STACK_REPLAY: items
    luisa::vector<ConditionStackItem> cond_stack;

    // Factory methods
    [[nodiscard]] static CoroInstruction make_simple(Instruction *inst) noexcept;
    [[nodiscard]] static CoroInstruction make_condition_stack_replay(luisa::vector<ConditionStackItem> items) noexcept;
    [[nodiscard]] static CoroInstruction make_first_flag() noexcept;
    [[nodiscard]] static CoroInstruction make_skip_if_first(CoroInstrRef flag, luisa::vector<CoroInstrRef> body) noexcept;
    [[nodiscard]] static CoroInstruction make_clear_first(CoroInstrRef flag) noexcept;
    [[nodiscard]] static CoroInstruction make_loop(luisa::vector<CoroInstrRef> prepare,
                                                    luisa::vector<CoroInstrRef> body,
                                                    luisa::vector<CoroInstrRef> update) noexcept;
    [[nodiscard]] static CoroInstruction make_if(CoroInstrRef cond,
                                                  luisa::vector<CoroInstrRef> true_branch,
                                                  luisa::vector<CoroInstrRef> false_branch) noexcept;
    [[nodiscard]] static CoroInstruction make_switch(CoroInstrRef value,
                                                      luisa::vector<CoroSwitchCase> cases,
                                                      luisa::vector<CoroInstrRef> default_branch) noexcept;
    [[nodiscard]] static CoroInstruction make_suspend(uint32_t token) noexcept;
    [[nodiscard]] static CoroInstruction make_terminate() noexcept;
};

// A scope in the CoroGraph: either the entry scope or a continuation.
struct CoroScope {
    luisa::vector<CoroInstrRef> instructions;
};

// The CoroGraph: result of CFG distillation and scope splitting.
// Each scope corresponds to either the entry subroutine or a continuation
// that resumes after a specific suspend token.
class LUISA_XIR_API CoroGraph {

public:
    luisa::vector<CoroInstruction> instructions;
    luisa::vector<CoroScope> scopes;
    CoroScopeRef entry{coro_invalid_ref};
    luisa::unordered_map<uint32_t, CoroScopeRef> tokens;

    // Add an instruction to the pool, return its ref.
    [[nodiscard]] CoroInstrRef add_instruction(CoroInstruction inst) noexcept;

    // Add a scope, return its ref.
    [[nodiscard]] CoroScopeRef add_scope(CoroScope scope) noexcept;

    // Build the CoroGraph from a coroutine function.
    // The function must have been through coro_normalize_cf_pass first.
    [[nodiscard]] static CoroGraph build(FunctionDefinition *function) noexcept;
};

}// namespace luisa::compute::xir
