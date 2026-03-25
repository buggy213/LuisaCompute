#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/value.h>

namespace luisa::compute {
class Type;
}

namespace luisa::compute::xir {

// An index into an access chain. Static indices are known at compile time
// (e.g., struct field access), while dynamic indices represent runtime values
// (e.g., array indexing with a variable).
struct AccessChainIndex {
    enum Tag { STATIC, DYNAMIC };
    Tag tag;
    union {
        int32_t static_index;
        Value *dynamic_value;
    };

    static AccessChainIndex make_static(int32_t index) noexcept {
        AccessChainIndex r;
        r.tag = STATIC;
        r.static_index = index;
        return r;
    }
    static AccessChainIndex make_dynamic(Value *v) noexcept {
        AccessChainIndex r;
        r.tag = DYNAMIC;
        r.dynamic_value = v;
        return r;
    }

    [[nodiscard]] bool is_static() const noexcept { return tag == STATIC; }
    [[nodiscard]] bool is_dynamic() const noexcept { return tag == DYNAMIC; }

    [[nodiscard]] bool operator==(const AccessChainIndex &other) const noexcept {
        if (tag != other.tag) return false;
        return tag == STATIC ? static_index == other.static_index :
                               dynamic_value == other.dynamic_value;
    }
};

struct AccessChainIndexHash {
    [[nodiscard]] size_t operator()(const AccessChainIndex &idx) const noexcept {
        if (idx.tag == AccessChainIndex::STATIC) {
            return std::hash<int32_t>{}(idx.static_index) ^ 0x9e3779b9u;
        }
        return std::hash<void *>{}(idx.dynamic_value) ^ 0x517cc1b7u;
    }
};

// The AccessTree tracks which (sub-)members of IR values are accessed.
// Each root in the tree corresponds to an IR Value* (typically an AllocaInst).
// Children represent sub-member accesses (struct fields, vector elements, etc.).
// A leaf node (empty children map) means the value is accessed as a whole at that level.
//
// This data structure is central to member-wise coroutine frame analysis:
// it allows tracking exactly which scalar fields of aggregates need to be
// saved/loaded across suspension points.
class LUISA_XIR_API AccessTree {

public:
    using NodeRef = uint32_t;
    static constexpr NodeRef invalid_node = ~0u;

    struct Node {
        luisa::unordered_map<AccessChainIndex, NodeRef, AccessChainIndexHash> children;
    };

private:
    luisa::unordered_map<Value *, NodeRef> _roots;
    luisa::vector<Node> _storage;

public:
    AccessTree() noexcept = default;

    // Add a new internal node, returns its index
    [[nodiscard]] NodeRef add_node() noexcept;

    // Access the node storage
    [[nodiscard]] Node &node(NodeRef ref) noexcept;
    [[nodiscard]] const Node &node(NodeRef ref) const noexcept;

    // Root access
    [[nodiscard]] auto &roots() noexcept { return _roots; }
    [[nodiscard]] auto &roots() const noexcept { return _roots; }
    [[nodiscard]] bool has_root(Value *v) const noexcept;
    [[nodiscard]] NodeRef root(Value *v) const noexcept;

    // Insert: mark that `value` is accessed at the given access chain.
    // An empty chain means the value is accessed as a whole.
    void insert(Value *value, luisa::span<const AccessChainIndex> chain) noexcept;

    // Check if the value is fully contained (accessed as a whole at this chain)
    [[nodiscard]] bool contains(Value *value, luisa::span<const AccessChainIndex> chain) const noexcept;

    // Check if the value maybe overlaps with the given chain (conservative)
    [[nodiscard]] bool maybe_overlaps(Value *value, luisa::span<const AccessChainIndex> chain) const noexcept;

    // Check if the tree is empty
    [[nodiscard]] bool empty() const noexcept { return _roots.empty(); }

    // Set operations (return new trees)
    [[nodiscard]] AccessTree intersect(const AccessTree &other) const noexcept;
    [[nodiscard]] AccessTree union_with(const AccessTree &other) const noexcept;
    [[nodiscard]] AccessTree subtract(const AccessTree &other) const noexcept;

    // Optimize: if all children of a composite are leaf nodes, collapse to a single leaf
    void coalesce_whole_access_chains() noexcept;

    // Collapse dynamic access chains to whole-access
    void dynamic_access_chains_as_whole() noexcept;

private:
    [[nodiscard]] NodeRef _clone_node(const AccessTree &src, NodeRef src_ref) noexcept;
    [[nodiscard]] NodeRef _intersect_nodes(const AccessTree &a, NodeRef a_ref,
                                            const AccessTree &b, NodeRef b_ref) noexcept;
    [[nodiscard]] NodeRef _union_nodes(const AccessTree &a, NodeRef a_ref,
                                       const AccessTree &b, NodeRef b_ref) noexcept;
    void _coalesce(NodeRef ref, const Type *type) noexcept;
    void _collapse_dynamic(NodeRef ref, const Type *type) noexcept;
    void _enumerate_for_subtraction(Value *value, const Type *type,
                                     const Node &access_node,
                                     luisa::vector<AccessChainIndex> &chain,
                                     const AccessTree &other,
                                     AccessTree &result) const noexcept;
    void _enumerate_all_children_for_subtraction(Value *value, const Type *type,
                                                  luisa::vector<AccessChainIndex> &chain,
                                                  const AccessTree &other,
                                                  AccessTree &result) const noexcept;
};

}// namespace luisa::compute::xir
