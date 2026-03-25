#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Coroutine suspension point. This instruction is a terminator that
// transfers control back to the coroutine scheduler. Each suspend
// point is assigned a unique token during the CoroGraph analysis,
// which identifies the corresponding continuation subroutine.
class LUISA_XIR_API SuspendInst final : public PrintMessageMixin<DerivedTerminatorInstruction<SuspendInst, DerivedInstructionTag::SUSPEND>> {

private:
    uint32_t _token{~0u};

public:
    // tag is the optional user-specified mark name (e.g., $suspend("intersect"))
    explicit SuspendInst(BasicBlock *parent_block, luisa::string tag = {}) noexcept;

    void set_token(uint32_t token) noexcept;
    [[nodiscard]] auto token() const noexcept -> uint32_t;

    [[nodiscard]] SuspendInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
