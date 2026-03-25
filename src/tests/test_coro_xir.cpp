// test playground for coroutine transformation

#include <luisa/luisa-compute.h>
#include <luisa/xir/passes/coro_normalize_cf.h>

using namespace luisa;
using namespace luisa::compute;

int main() {
    Callable trivial = []() noexcept {
        Float a = 1.0f;
        Float b = 2.0f;
        // manually insert suspend here
        Float c = a + b;
        device_log("c = %f", c);
    };

    auto module = xir::ast_to_xir_translate(trivial.function(), {});
    auto text = xir::xir_to_text_translate(module.get(), true);
    LUISA_INFO("AST2IR:\n{}", text);

    auto manual_module = luisa::make_unique<xir::Module>();

    // this seems a bit strange to me, why doesn't it just contain the data directly for scalar types?
    float one_data = 1.0f;
    xir::Constant *one = manual_module->create_constant(Type::of<float>(), &one_data);
    float two_data = 2.0f;
    xir::Constant *two = manual_module->create_constant(Type::of<float>(), &two_data);

    xir::FunctionDefinition* fn = manual_module->create_callable(nullptr); // corresponds to void return type
    xir::BasicBlock* body = fn->create_body_block();

    xir::XIRBuilder builder;
    builder.set_insertion_point(body);

    xir::AllocaInst* a_decl = builder.alloca_local(Type::of<float>());
    xir::AllocaInst* b_decl = builder.alloca_local(Type::of<float>());
    xir::AllocaInst* c_decl = builder.alloca_local(Type::of<float>());
    builder.store(a_decl, one);
    auto after_store = builder.store(b_decl, two);
    xir::LoadInst* a_load = builder.load(Type::of<float>(), a_decl);
    xir::LoadInst* b_load = builder.load(Type::of<float>(), b_decl);
    std::array<xir::Value*, 2> operands = { a_load, b_load };
    xir::ArithmeticInst* a_plus_b = builder.call(Type::of<float>(), xir::ArithmeticOp::BINARY_ADD, operands);
    builder.store(c_decl, a_plus_b);
    xir::LoadInst* c_load = builder.load(Type::of<float>(), c_decl);
    builder.print("c = %f", { c_load });
    builder.return_void();

    text = xir::xir_to_text_translate(manual_module.get(), true);
    LUISA_INFO("manual IR:\n{}", text);

    // apply transform to XIR
    fn->set_coroutine(true);
    builder.set_insertion_point(after_store);
    builder.suspend_();

    text = xir::xir_to_text_translate(manual_module.get(), true);
    LUISA_INFO("manual IR with suspend:\n{}", text);

    xir::coro_normalize_cf_pass_run_on_module(manual_module.get());


}