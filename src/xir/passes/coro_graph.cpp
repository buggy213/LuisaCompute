#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/suspend.h>
#include <luisa/xir/passes/coro_graph.h>

namespace luisa::compute::xir {

// --- CoroInstruction factory methods ---

CoroInstruction CoroInstruction::make_simple(Instruction *inst) noexcept {
    CoroInstruction ci{};
    ci.tag = SIMPLE;
    ci.ir_inst = inst;
    return ci;
}

CoroInstruction CoroInstruction::make_condition_stack_replay(luisa::vector<ConditionStackItem> items) noexcept {
    CoroInstruction ci{};
    ci.tag = CONDITION_STACK_REPLAY;
    ci.cond_stack = std::move(items);
    return ci;
}

CoroInstruction CoroInstruction::make_first_flag() noexcept {
    CoroInstruction ci{};
    ci.tag = MAKE_FIRST_FLAG;
    return ci;
}

CoroInstruction CoroInstruction::make_skip_if_first(CoroInstrRef flag, luisa::vector<CoroInstrRef> body) noexcept {
    CoroInstruction ci{};
    ci.tag = SKIP_IF_FIRST_FLAG;
    ci.ref = flag;
    ci.body = std::move(body);
    return ci;
}

CoroInstruction CoroInstruction::make_clear_first(CoroInstrRef flag) noexcept {
    CoroInstruction ci{};
    ci.tag = CLEAR_FIRST_FLAG;
    ci.ref = flag;
    return ci;
}

CoroInstruction CoroInstruction::make_loop(luisa::vector<CoroInstrRef> prepare,
                                            luisa::vector<CoroInstrRef> body,
                                            luisa::vector<CoroInstrRef> update) noexcept {
    CoroInstruction ci{};
    ci.tag = LOOP;
    ci.prepare = std::move(prepare);
    ci.body = std::move(body);
    ci.update = std::move(update);
    return ci;
}

CoroInstruction CoroInstruction::make_if(CoroInstrRef cond,
                                          luisa::vector<CoroInstrRef> true_branch,
                                          luisa::vector<CoroInstrRef> false_branch) noexcept {
    CoroInstruction ci{};
    ci.tag = IF;
    ci.ref = cond;
    ci.body = std::move(true_branch);
    ci.else_body = std::move(false_branch);
    return ci;
}

CoroInstruction CoroInstruction::make_switch(CoroInstrRef value,
                                              luisa::vector<CoroSwitchCase> cases,
                                              luisa::vector<CoroInstrRef> default_branch) noexcept {
    CoroInstruction ci{};
    ci.tag = SWITCH;
    ci.ref = value;
    ci.cases = std::move(cases);
    ci.else_body = std::move(default_branch);
    return ci;
}

CoroInstruction CoroInstruction::make_suspend(uint32_t token) noexcept {
    CoroInstruction ci{};
    ci.tag = SUSPEND;
    ci.token = token;
    return ci;
}

CoroInstruction CoroInstruction::make_terminate() noexcept {
    CoroInstruction ci{};
    ci.tag = TERMINATE;
    return ci;
}

// --- CoroGraph methods ---

CoroInstrRef CoroGraph::add_instruction(CoroInstruction inst) noexcept {
    auto ref = static_cast<CoroInstrRef>(instructions.size());
    instructions.emplace_back(std::move(inst));
    return ref;
}

CoroScopeRef CoroGraph::add_scope(CoroScope scope) noexcept {
    auto ref = static_cast<CoroScopeRef>(scopes.size());
    scopes.emplace_back(std::move(scope));
    return ref;
}

// =============================================================================
// Preliminary graph: translates structured XIR into flat CoroInstruction nodes.
// =============================================================================

namespace {

struct PreliminaryGraph {
    luisa::vector<CoroInstruction> instructions;
    luisa::vector<CoroInstrRef> entry_body;
    luisa::unordered_map<Instruction *, CoroInstrRef> ir_to_coro;
    luisa::unordered_map<CoroInstrRef, bool> terminators;// instructions that definitely terminate

    CoroInstrRef add(CoroInstruction inst) noexcept {
        auto ref = static_cast<CoroInstrRef>(instructions.size());
        instructions.emplace_back(std::move(inst));
        return ref;
    }

    // Translate a Value (which may or may not be an Instruction) into a CoroInstrRef.
    // If the value is an Instruction already in the cache, returns its ref.
    // Otherwise creates a SIMPLE wrapper.
    CoroInstrRef translate_value(Value *v) noexcept {
        if (v == nullptr) return coro_invalid_ref;
        if (v->isa<Instruction>()) {
            return translate_instruction(static_cast<Instruction *>(v));
        }
        // Non-instruction value (Argument, Constant, etc.)
        // Create a SIMPLE node with nullptr ir_inst to represent it
        return add(CoroInstruction::make_simple(nullptr));
    }

    // Translate a basic block's instructions into CoroInstrRefs.
    luisa::vector<CoroInstrRef> translate_block(BasicBlock *block) noexcept {
        luisa::vector<CoroInstrRef> result;
        if (block == nullptr) return result;
        block->traverse_instructions([&](Instruction *inst) noexcept {
            auto ref = translate_instruction(inst);
            if (ref != coro_invalid_ref) {
                result.emplace_back(ref);
            }
        });
        return result;
    }

    // Translate a single IR instruction into a CoroInstruction.
    CoroInstrRef translate_instruction(Instruction *inst) noexcept {
        // Check cache
        if (auto it = ir_to_coro.find(inst); it != ir_to_coro.end()) {
            return it->second;
        }

        CoroInstrRef ref;
        if (inst->isa<SuspendInst>()) {
            auto suspend = static_cast<SuspendInst *>(inst);
            ref = add(CoroInstruction::make_suspend(suspend->token()));
        } else if (inst->isa<ReturnInst>()) {
            ref = add(CoroInstruction::make_terminate());
        } else if (inst->isa<IfInst>()) {
            auto if_inst = static_cast<IfInst *>(inst);
            auto cond = translate_value(if_inst->condition());
            auto true_branch = translate_block(if_inst->true_block());
            auto false_branch = translate_block(if_inst->false_block());
            ref = add(CoroInstruction::make_if(cond, std::move(true_branch), std::move(false_branch)));
        } else if (inst->isa<SwitchInst>()) {
            auto switch_inst = static_cast<SwitchInst *>(inst);
            auto value = translate_value(switch_inst->value());
            luisa::vector<CoroSwitchCase> cases;
            auto n_cases = switch_inst->case_count();
            for (size_t i = 0; i < n_cases; i++) {
                auto body = translate_block(switch_inst->case_block(i));
                cases.push_back({switch_inst->case_value(i), std::move(body)});
            }
            auto default_body = translate_block(switch_inst->default_block());
            ref = add(CoroInstruction::make_switch(value, std::move(cases), std::move(default_body)));
        } else if (inst->isa<LoopInst>()) {
            auto loop_inst = static_cast<LoopInst *>(inst);
            auto prepare = translate_block(loop_inst->prepare_block());
            auto body = translate_block(loop_inst->body_block());
            auto update = translate_block(loop_inst->update_block());
            ref = add(CoroInstruction::make_loop(std::move(prepare), std::move(body), std::move(update)));
        } else {
            // All other instructions are simple nodes
            ref = add(CoroInstruction::make_simple(inst));
        }

        ir_to_coro.emplace(inst, ref);
        return ref;
    }

    // DFS to find which instructions definitely terminate control flow.
    bool compute_terminates(CoroInstrRef ref) noexcept {
        auto &ci = instructions[ref];
        bool result = false;
        switch (ci.tag) {
            case CoroInstruction::TERMINATE:
            case CoroInstruction::SUSPEND:
                result = true;
                break;
            case CoroInstruction::IF:
                result = block_terminates(ci.body) && block_terminates(ci.else_body);
                break;
            case CoroInstruction::SWITCH: {
                result = block_terminates(ci.else_body);
                for (auto &c : ci.cases) {
                    if (!block_terminates(c.body)) {
                        result = false;
                        break;
                    }
                }
                break;
            }
            case CoroInstruction::LOOP:
                // A loop only terminates if its body always terminates
                // (without going through the loop condition)
                result = block_terminates(ci.body);
                break;
            default:
                result = false;
                break;
        }
        if (result) {
            terminators.emplace(ref, true);
        }
        return result;
    }

    bool block_terminates(const luisa::vector<CoroInstrRef> &block) noexcept {
        for (auto ref : block) {
            if (compute_terminates(ref)) return true;
        }
        return false;
    }

    void find_all_terminators() noexcept {
        block_terminates(entry_body);
    }
};

// =============================================================================
// Continuation extraction: scope splitting at suspend points.
// =============================================================================

// Position within the CoroGraph instruction tree.
struct GraphPosition {
    CoroInstrRef parent;     // the parent CF instruction (IF/SWITCH/LOOP)
    uint32_t branch;         // which branch of parent (0=true/body/prepare, 1=false/default, 2+=case/update)
    uint32_t index;          // position within that branch
    bool inside_loop;        // whether this position is inside a loop
};

struct ContinuationExtractor {
    CoroGraph &graph;
    const PreliminaryGraph &prelim;

    // Get a ref to the block array for a given position's parent+branch.
    const luisa::vector<CoroInstrRef> &get_branch(CoroInstrRef parent, uint32_t branch) const noexcept {
        auto &ci = graph.instructions[parent];
        switch (ci.tag) {
            case CoroInstruction::IF:
                return branch == 0 ? ci.body : ci.else_body;
            case CoroInstruction::SWITCH:
                if (branch < ci.cases.size()) return ci.cases[branch].body;
                return ci.else_body;// default
            case CoroInstruction::LOOP:
                if (branch == 0) return ci.prepare;
                if (branch == 1) return ci.body;
                return ci.update;// branch == 2
            case CoroInstruction::SKIP_IF_FIRST_FLAG:
                return ci.body;
            default:
                LUISA_ERROR_WITH_LOCATION("Invalid parent type for get_branch.");
        }
    }

    // Check if a block starting at `start_idx` terminates.
    bool block_terminates_from(const luisa::vector<CoroInstrRef> &block, uint32_t start_idx) const noexcept {
        for (uint32_t i = start_idx; i < block.size(); i++) {
            if (prelim.terminators.contains(block[i])) return true;
        }
        return false;
    }

    // Clone an instruction and its sub-blocks (deep copy).
    // Note: we snapshot fields before recursing because clone_block may call
    // graph.add_instruction, which can reallocate graph.instructions and
    // invalidate any reference into it.
    CoroInstrRef clone_instruction(CoroInstrRef ref) noexcept {
        auto tag = graph.instructions[ref].tag;
        switch (tag) {
            case CoroInstruction::SIMPLE:
            case CoroInstruction::SUSPEND:
            case CoroInstruction::TERMINATE:
            case CoroInstruction::MAKE_FIRST_FLAG:
            case CoroInstruction::CLEAR_FIRST_FLAG:
            case CoroInstruction::CONDITION_STACK_REPLAY:
                return ref;
            case CoroInstruction::IF: {
                auto cond_ref = graph.instructions[ref].ref;
                auto true_src = graph.instructions[ref].body;
                auto false_src = graph.instructions[ref].else_body;
                auto new_true = clone_block(true_src);
                auto new_false = clone_block(false_src);
                return graph.add_instruction(CoroInstruction::make_if(cond_ref, std::move(new_true), std::move(new_false)));
            }
            case CoroInstruction::SWITCH: {
                auto value_ref = graph.instructions[ref].ref;
                auto cases_src = graph.instructions[ref].cases;
                auto default_src = graph.instructions[ref].else_body;
                luisa::vector<CoroSwitchCase> new_cases;
                for (auto &c : cases_src) {
                    new_cases.push_back({c.value, clone_block(c.body)});
                }
                auto new_default = clone_block(default_src);
                return graph.add_instruction(CoroInstruction::make_switch(value_ref, std::move(new_cases), std::move(new_default)));
            }
            case CoroInstruction::LOOP: {
                auto prepare_src = graph.instructions[ref].prepare;
                auto body_src = graph.instructions[ref].body;
                auto update_src = graph.instructions[ref].update;
                auto new_prepare = clone_block(prepare_src);
                auto new_body = clone_block(body_src);
                auto new_update = clone_block(update_src);
                return graph.add_instruction(CoroInstruction::make_loop(std::move(new_prepare), std::move(new_body), std::move(new_update)));
            }
            case CoroInstruction::SKIP_IF_FIRST_FLAG: {
                auto flag_ref = graph.instructions[ref].ref;
                auto body_src = graph.instructions[ref].body;
                auto new_body = clone_block(body_src);
                return graph.add_instruction(CoroInstruction::make_skip_if_first(flag_ref, std::move(new_body)));
            }
            default:
                LUISA_ERROR_WITH_LOCATION("Unexpected instruction type in clone.");
        }
    }

    // Clone a block of instructions, stopping at the first terminator.
    luisa::vector<CoroInstrRef> clone_block(const luisa::vector<CoroInstrRef> &block) noexcept {
        luisa::vector<CoroInstrRef> result;
        for (auto ref : block) {
            result.emplace_back(clone_instruction(ref));
            if (prelim.terminators.contains(ref)) break;
        }
        return result;
    }

    // Find the ancestor chain from a suspend point back to the entry.
    // Returns ancestors from outermost to innermost.
    luisa::vector<GraphPosition> find_reachable_ancestors(
        const luisa::vector<GraphPosition> &ancestor_stack) const noexcept {
        luisa::vector<GraphPosition> reachable;
        // Walk ancestors in reverse (innermost to outermost)
        for (int i = static_cast<int>(ancestor_stack.size()) - 1; i >= 0; i--) {
            auto &pos = ancestor_stack[i];
            auto &branch = get_branch(pos.parent, pos.branch);
            // Check if the block from the NEXT instruction onward terminates early
            if (block_terminates_from(branch, pos.index + 1)) {
                // Remaining code terminates, so outer ancestors are unreachable
                reachable.emplace_back(pos);
                break;
            }
            reachable.emplace_back(pos);
        }
        // Reverse to get outermost→innermost order
        std::reverse(reachable.begin(), reachable.end());
        return reachable;
    }

    // Build the condition stack for replay at the start of a continuation.
    luisa::vector<ConditionStackItem> build_condition_stack(
        const luisa::vector<GraphPosition> &ancestors) const noexcept {
        luisa::vector<ConditionStackItem> items;
        for (auto &pos : ancestors) {
            auto &ci = graph.instructions[pos.parent];
            if (ci.tag == CoroInstruction::IF) {
                // The condition instruction
                auto cond_ref = ci.ref;
                auto &cond_inst = graph.instructions[cond_ref];
                if (cond_inst.tag == CoroInstruction::SIMPLE && cond_inst.ir_inst != nullptr) {
                    items.push_back({cond_inst.ir_inst, static_cast<int32_t>(pos.branch == 0 ? 1 : 0)});
                }
            } else if (ci.tag == CoroInstruction::SWITCH) {
                auto cond_ref = ci.ref;
                auto &cond_inst = graph.instructions[cond_ref];
                if (pos.branch < ci.cases.size()) {
                    // Named case, record value
                    if (cond_inst.tag == CoroInstruction::SIMPLE && cond_inst.ir_inst != nullptr) {
                        items.push_back({cond_inst.ir_inst, ci.cases[pos.branch].value});
                    }
                }
                // Default branch: don't add to condition stack (value is unknown)
            }
            // LOOP: no condition to replay (loop always re-enters on resume)
        }
        return items;
    }

    // Construct a continuation scope for a suspend point.
    // `ancestors` is the reachable ancestor chain (outermost → innermost).
    // `suspend_branch` and `suspend_index` identify the suspend within its immediate parent.
    CoroScope construct_subscope(
        const luisa::vector<GraphPosition> &ancestors,
        const luisa::vector<CoroInstrRef> &suspend_block,
        uint32_t suspend_index) noexcept {

        CoroScope scope;

        // Step 1: Condition stack replay
        auto cond_stack = build_condition_stack(ancestors);
        if (!cond_stack.empty()) {
            auto replay_ref = graph.add_instruction(
                CoroInstruction::make_condition_stack_replay(std::move(cond_stack)));
            scope.instructions.emplace_back(replay_ref);
        }

        // Step 2: First flag for SkipIfFirst mechanism
        auto first_flag = graph.add_instruction(CoroInstruction::make_first_flag());
        scope.instructions.emplace_back(first_flag);

        // Step 3: Reconstruct the control flow tree
        // Start from the outermost ancestor and work inward
        auto inner_body = construct_ancestor_chain(ancestors, 0, first_flag,
                                                    suspend_block, suspend_index);
        for (auto &ref : inner_body) {
            scope.instructions.emplace_back(ref);
        }

        // Step 4: Append remaining instructions in the outermost scope
        // (instructions after the outermost ancestor's parent)
        if (!ancestors.empty()) {
            auto &outermost = ancestors[0];
            auto &outer_branch = get_branch(outermost.parent, outermost.branch);
            // Instructions after the outermost ancestor's position
            for (uint32_t i = outermost.index + 1; i < outer_branch.size(); i++) {
                scope.instructions.emplace_back(clone_instruction(outer_branch[i]));
                if (prelim.terminators.contains(outer_branch[i])) break;
            }
        }

        // Step 5: Remove unreachable code from scope
        remove_unreachable(scope.instructions);

        return scope;
    }

    // Recursively reconstruct the ancestor chain for continuation extraction.
    // Returns a list of instructions for this level.
    luisa::vector<CoroInstrRef> construct_ancestor_chain(
        const luisa::vector<GraphPosition> &ancestors,
        size_t ancestor_idx,
        CoroInstrRef first_flag,
        const luisa::vector<CoroInstrRef> &suspend_block,
        uint32_t suspend_index) noexcept {

        luisa::vector<CoroInstrRef> result;

        if (ancestor_idx >= ancestors.size()) {
            // Base case: we've reached the suspend point.
            // Instructions after the suspend in its block.
            for (uint32_t i = suspend_index + 1; i < suspend_block.size(); i++) {
                result.emplace_back(clone_instruction(suspend_block[i]));
                if (prelim.terminators.contains(suspend_block[i])) break;
            }

            // Add ClearFirstFlag at the point where suspension was
            result.insert(result.begin(),
                          graph.add_instruction(CoroInstruction::make_clear_first(first_flag)));
            return result;
        }

        auto &pos = ancestors[ancestor_idx];
        auto &parent_ci = graph.instructions[pos.parent];

        if (parent_ci.tag == CoroInstruction::IF) {
            // Reconstruct If: the branch containing the suspend gets the
            // recursive continuation, other branch is cloned as-is.
            auto suspend_branch = pos.branch;
            luisa::vector<CoroInstrRef> new_true, new_false;

            if (suspend_branch == 0) {
                // Suspend is in true branch
                new_true = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
                new_false = clone_block(parent_ci.else_body);
            } else {
                // Suspend is in false branch
                new_true = clone_block(parent_ci.body);
                new_false = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
            }

            // If we're inside a loop, wrap the non-suspend branch with SkipIfFirst
            if (pos.inside_loop) {
                if (suspend_branch == 0) {
                    // True branch has suspend; false branch is wrapped
                    wrap_with_skip_if_first(new_false, first_flag);
                    wrap_with_skip_if_first(new_true, first_flag, true);
                } else {
                    wrap_with_skip_if_first(new_true, first_flag);
                    wrap_with_skip_if_first(new_false, first_flag, true);
                }
            }

            auto if_ref = graph.add_instruction(CoroInstruction::make_if(
                parent_ci.ref, std::move(new_true), std::move(new_false)));
            result.emplace_back(if_ref);

        } else if (parent_ci.tag == CoroInstruction::SWITCH) {
            auto suspend_branch = pos.branch;

            luisa::vector<CoroSwitchCase> new_cases;
            for (size_t i = 0; i < parent_ci.cases.size(); i++) {
                if (i == suspend_branch) {
                    auto body = construct_branch_with_continuation(
                        ancestors, ancestor_idx, first_flag, pos,
                        suspend_block, suspend_index);
                    if (pos.inside_loop) {
                        wrap_with_skip_if_first(body, first_flag, true);
                    }
                    new_cases.push_back({parent_ci.cases[i].value, std::move(body)});
                } else {
                    auto body = clone_block(parent_ci.cases[i].body);
                    if (pos.inside_loop) {
                        wrap_with_skip_if_first(body, first_flag);
                    }
                    new_cases.push_back({parent_ci.cases[i].value, std::move(body)});
                }
            }
            // Default branch
            luisa::vector<CoroInstrRef> new_default;
            if (suspend_branch >= parent_ci.cases.size()) {
                new_default = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
                if (pos.inside_loop) {
                    wrap_with_skip_if_first(new_default, first_flag, true);
                }
            } else {
                new_default = clone_block(parent_ci.else_body);
                if (pos.inside_loop) {
                    wrap_with_skip_if_first(new_default, first_flag);
                }
            }

            auto switch_ref = graph.add_instruction(CoroInstruction::make_switch(
                parent_ci.ref, std::move(new_cases), std::move(new_default)));
            result.emplace_back(switch_ref);

        } else if (parent_ci.tag == CoroInstruction::LOOP) {
            // Suspend is inside the loop. Reconstruct the loop with
            // SkipIfFirst guards on pre-suspend instructions.
            auto suspend_sub_branch = pos.branch;// 0=prepare, 1=body, 2=update

            // Clone loop blocks, with the suspend branch getting the continuation
            luisa::vector<CoroInstrRef> new_prepare, new_body, new_update;

            if (suspend_sub_branch == 0) {
                new_prepare = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
                new_body = clone_block(parent_ci.body);
                new_update = clone_block(parent_ci.update);
            } else if (suspend_sub_branch == 1) {
                new_prepare = clone_block(parent_ci.prepare);
                new_body = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
                new_update = clone_block(parent_ci.update);
            } else {
                new_prepare = clone_block(parent_ci.prepare);
                new_body = clone_block(parent_ci.body);
                new_update = construct_branch_with_continuation(
                    ancestors, ancestor_idx, first_flag, pos,
                    suspend_block, suspend_index);
            }

            auto loop_ref = graph.add_instruction(CoroInstruction::make_loop(
                std::move(new_prepare), std::move(new_body), std::move(new_update)));
            result.emplace_back(loop_ref);
        }

        return result;
    }

    // Construct a branch that contains the continuation (instructions after suspend
    // within this branch level + recursive continuation from inner ancestors).
    luisa::vector<CoroInstrRef> construct_branch_with_continuation(
        const luisa::vector<GraphPosition> &ancestors,
        size_t ancestor_idx,
        CoroInstrRef first_flag,
        const GraphPosition &pos,
        const luisa::vector<CoroInstrRef> &suspend_block,
        uint32_t suspend_index) noexcept {

        luisa::vector<CoroInstrRef> result;

        // Instructions before the inner ancestor/suspend in this branch
        auto &branch = get_branch(pos.parent, pos.branch);
        for (uint32_t i = 0; i < pos.index; i++) {
            result.emplace_back(clone_instruction(branch[i]));
        }

        // If inside a loop and there are pre-suspend instructions, wrap with SkipIfFirst
        if (pos.inside_loop && pos.index > 0) {
            luisa::vector<CoroInstrRef> pre_suspend(result.begin(), result.end());
            result.clear();
            auto skip_ref = graph.add_instruction(
                CoroInstruction::make_skip_if_first(first_flag, std::move(pre_suspend)));
            result.emplace_back(skip_ref);
        }

        // The inner ancestor/suspend instruction itself and continuation
        auto inner = construct_ancestor_chain(ancestors, ancestor_idx + 1,
                                               first_flag, suspend_block, suspend_index);
        for (auto &ref : inner) {
            result.emplace_back(ref);
        }

        // Instructions after the inner ancestor in this branch
        for (uint32_t i = pos.index + 1; i < branch.size(); i++) {
            result.emplace_back(clone_instruction(branch[i]));
            if (prelim.terminators.contains(branch[i])) break;
        }

        return result;
    }

    // Wrap a block with SkipIfFirst. If `has_continuation` is true,
    // only wrap the pre-continuation part.
    void wrap_with_skip_if_first(luisa::vector<CoroInstrRef> &block,
                                  CoroInstrRef first_flag,
                                  bool has_continuation = false) noexcept {
        if (block.empty()) return;
        if (!has_continuation) {
            // Wrap entire block
            auto skip_ref = graph.add_instruction(
                CoroInstruction::make_skip_if_first(first_flag, std::move(block)));
            block.clear();
            block.emplace_back(skip_ref);
        }
        // If has_continuation, the block already has SkipIfFirst from
        // construct_branch_with_continuation, don't double-wrap.
    }

    // Remove unreachable instructions from a block (after a terminator).
    void remove_unreachable(luisa::vector<CoroInstrRef> &block) noexcept {
        for (size_t i = 0; i < block.size(); i++) {
            if (prelim.terminators.contains(block[i])) {
                // Special case: if terminator is a loop, keep it but
                // remove everything after
                block.resize(i + 1);
                return;
            }
        }
    }

    // Main entry point: traverse the preliminary graph and extract
    // continuations at each suspend point.
    void extract_all(const luisa::vector<CoroInstrRef> &entry_body) noexcept {
        // The entry scope is the full function body
        CoroScope entry_scope;
        for (auto ref : entry_body) {
            entry_scope.instructions.emplace_back(ref);
        }
        graph.entry = graph.add_scope(std::move(entry_scope));

        // DFS traversal to find all suspend points
        luisa::vector<GraphPosition> ancestor_stack;
        traverse_for_suspends(entry_body, ancestor_stack);
    }

    // DFS traversal that finds suspend points and extracts continuations.
    void traverse_for_suspends(
        const luisa::vector<CoroInstrRef> &block,
        luisa::vector<GraphPosition> &ancestor_stack) noexcept {

        for (uint32_t i = 0; i < block.size(); i++) {
            auto ref = block[i];
            auto &ci = graph.instructions[ref];

            if (ci.tag == CoroInstruction::SUSPEND) {
                extract_continuation(ci.token, ancestor_stack, block, i);
            } else if (ci.tag == CoroInstruction::IF) {
                // Determine if we're inside a loop
                bool in_loop = std::any_of(ancestor_stack.begin(), ancestor_stack.end(),
                                            [this](const GraphPosition &p) {
                                                return graph.instructions[p.parent].tag == CoroInstruction::LOOP;
                                            });
                // Traverse true branch
                ancestor_stack.push_back({ref, 0, i, in_loop});
                traverse_for_suspends(ci.body, ancestor_stack);
                ancestor_stack.pop_back();
                // Traverse false branch
                ancestor_stack.push_back({ref, 1, i, in_loop});
                traverse_for_suspends(ci.else_body, ancestor_stack);
                ancestor_stack.pop_back();
            } else if (ci.tag == CoroInstruction::SWITCH) {
                bool in_loop = std::any_of(ancestor_stack.begin(), ancestor_stack.end(),
                                            [this](const GraphPosition &p) {
                                                return graph.instructions[p.parent].tag == CoroInstruction::LOOP;
                                            });
                // Traverse cases
                for (uint32_t c = 0; c < ci.cases.size(); c++) {
                    ancestor_stack.push_back({ref, c, i, in_loop});
                    traverse_for_suspends(ci.cases[c].body, ancestor_stack);
                    ancestor_stack.pop_back();
                }
                // Traverse default
                ancestor_stack.push_back({ref, static_cast<uint32_t>(ci.cases.size()), i, in_loop});
                traverse_for_suspends(ci.else_body, ancestor_stack);
                ancestor_stack.pop_back();
            } else if (ci.tag == CoroInstruction::LOOP) {
                // Traverse prepare
                ancestor_stack.push_back({ref, 0, i, true});
                traverse_for_suspends(ci.prepare, ancestor_stack);
                ancestor_stack.pop_back();
                // Traverse body
                ancestor_stack.push_back({ref, 1, i, true});
                traverse_for_suspends(ci.body, ancestor_stack);
                ancestor_stack.pop_back();
                // Traverse update
                ancestor_stack.push_back({ref, 2, i, true});
                traverse_for_suspends(ci.update, ancestor_stack);
                ancestor_stack.pop_back();
            }
        }
    }

    // Extract a continuation scope for a specific suspend token.
    void extract_continuation(
        uint32_t token,
        const luisa::vector<GraphPosition> &ancestor_stack,
        const luisa::vector<CoroInstrRef> &suspend_block,
        uint32_t suspend_index) noexcept {

        // Don't extract duplicate tokens
        if (graph.tokens.contains(token)) return;

        auto ancestors = find_reachable_ancestors(ancestor_stack);
        auto scope = construct_subscope(ancestors, suspend_block, suspend_index);
        auto scope_ref = graph.add_scope(std::move(scope));
        graph.tokens.emplace(token, scope_ref);
    }
};

}// anonymous namespace

// =============================================================================
// CoroGraph::build
// =============================================================================

CoroGraph CoroGraph::build(FunctionDefinition *function) noexcept {
    LUISA_ASSERT(function != nullptr, "Null function.");
    LUISA_ASSERT(function->is_coroutine(), "Function is not a coroutine.");
    LUISA_ASSERT(function->body_block() != nullptr, "Coroutine has no body block.");

    // Phase 1: Build preliminary graph
    PreliminaryGraph prelim;
    prelim.entry_body = prelim.translate_block(function->body_block());
    prelim.find_all_terminators();

    // Validate: no duplicate suspend tokens
    luisa::unordered_map<uint32_t, bool> seen_tokens;
    for (auto &ci : prelim.instructions) {
        if (ci.tag == CoroInstruction::SUSPEND) {
            LUISA_ASSERT(ci.token != ~0u, "Suspend instruction has no token assigned.");
            LUISA_ASSERT(!seen_tokens.contains(ci.token),
                         "Duplicate suspend token: {}.", ci.token);
            seen_tokens.emplace(ci.token, true);
        }
    }

    // Phase 2: Build CoroGraph with scope splitting
    CoroGraph graph;
    // Copy all preliminary instructions into the graph
    graph.instructions = std::move(prelim.instructions);

    ContinuationExtractor extractor{graph, prelim};
    extractor.extract_all(prelim.entry_body);

    return graph;
}

}// namespace luisa::compute::xir
