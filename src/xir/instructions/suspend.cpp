#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/suspend.h>

namespace luisa::compute::xir {

SuspendInst::SuspendInst(BasicBlock *parent_block, luisa::string tag) noexcept
    : Super{std::move(tag), parent_block} {}

void SuspendInst::set_token(uint32_t token) noexcept {
    _token = token;
}

auto SuspendInst::token() const noexcept -> uint32_t {
    return _token;
}

SuspendInst *SuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto inst = b.suspend_(message());
    inst->set_token(_token);
    return inst;
}

}// namespace luisa::compute::xir
