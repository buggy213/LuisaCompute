#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/xir/passes/coro_access_tree.h>

namespace luisa::compute::xir {

AccessTree::NodeRef AccessTree::add_node() noexcept {
    auto ref = static_cast<NodeRef>(_storage.size());
    _storage.emplace_back();
    return ref;
}

AccessTree::Node &AccessTree::node(NodeRef ref) noexcept {
    return _storage[ref];
}

const AccessTree::Node &AccessTree::node(NodeRef ref) const noexcept {
    return _storage[ref];
}

bool AccessTree::has_root(Value *v) const noexcept {
    return _roots.contains(v);
}

AccessTree::NodeRef AccessTree::root(Value *v) const noexcept {
    auto it = _roots.find(v);
    return it != _roots.end() ? it->second : invalid_node;
}

void AccessTree::insert(Value *value, luisa::span<const AccessChainIndex> chain) noexcept {
    auto parent_is_new = false;
    NodeRef current;
    if (auto it = _roots.find(value); it != _roots.end()) {
        current = it->second;
    } else {
        current = add_node();
        _roots.emplace(value, current);
        parent_is_new = true;
    }

    for (auto &idx : chain) {
        auto &children = node(current).children;
        if (children.empty() && !parent_is_new) {
            // existing leaf node (accessed as a whole), no need to refine further
            break;
        }
        if (auto it = children.find(idx); it != children.end()) {
            current = it->second;
            parent_is_new = false;
        } else {
            auto new_node = add_node();
            children.emplace(idx, new_node);
            parent_is_new = true;
            current = new_node;
        }
    }
    // mark as whole access at this level
    node(current).children.clear();
}

bool AccessTree::contains(Value *value, luisa::span<const AccessChainIndex> chain) const noexcept {
    auto it = _roots.find(value);
    if (it == _roots.end()) return false;
    auto current = it->second;
    for (auto &idx : chain) {
        auto &children = node(current).children;
        if (children.empty()) {
            // accessed as a whole, so any sub-access is contained
            return true;
        }
        auto child_it = children.find(idx);
        if (child_it == children.end()) return false;
        current = child_it->second;
    }
    return node(current).children.empty();
}

bool AccessTree::maybe_overlaps(Value *value, luisa::span<const AccessChainIndex> chain) const noexcept {
    auto it = _roots.find(value);
    if (it == _roots.end()) return false;
    auto current = it->second;
    for (auto &idx : chain) {
        auto &children = node(current).children;
        if (children.empty()) return true;// whole access
        auto child_it = children.find(idx);
        if (child_it != children.end()) {
            current = child_it->second;
        } else if (idx.is_dynamic() ||
                   std::any_of(children.begin(), children.end(),
                               [](auto &p) { return p.first.is_dynamic(); })) {
            // dynamic indices: conservatively assume overlap
            return true;
        } else {
            return false;
        }
    }
    // chain is a prefix of something in the tree → overlaps
    return true;
}

AccessTree::NodeRef AccessTree::_clone_node(const AccessTree &src, NodeRef src_ref) noexcept {
    auto new_ref = add_node();
    auto &src_node = src.node(src_ref);
    for (auto &[idx, child_ref] : src_node.children) {
        node(new_ref).children.emplace(idx, _clone_node(src, child_ref));
    }
    return new_ref;
}

AccessTree::NodeRef AccessTree::_intersect_nodes(const AccessTree &a, NodeRef a_ref,
                                                  const AccessTree &b, NodeRef b_ref) noexcept {
    auto &a_node = a.node(a_ref);
    auto &b_node = b.node(b_ref);
    if (a_node.children.empty()) {
        // a is whole → intersection is b
        return _clone_node(b, b_ref);
    }
    if (b_node.children.empty()) {
        // b is whole → intersection is a
        return _clone_node(a, a_ref);
    }
    // find common children
    auto new_ref = add_node();
    for (auto &[idx, a_child] : a_node.children) {
        auto b_it = b_node.children.find(idx);
        if (b_it != b_node.children.end()) {
            auto child = _intersect_nodes(a, a_child, b, b_it->second);
            // only keep non-empty intersections
            if (!node(child).children.empty() || a.node(a_child).children.empty() || b.node(b_it->second).children.empty()) {
                node(new_ref).children.emplace(idx, child);
            }
        }
    }
    return new_ref;
}

AccessTree AccessTree::intersect(const AccessTree &other) const noexcept {
    AccessTree result;
    for (auto &[value, ref] : _roots) {
        auto other_it = other._roots.find(value);
        if (other_it != other._roots.end()) {
            auto merged = result._intersect_nodes(*this, ref, other, other_it->second);
            if (!result.node(merged).children.empty() ||
                node(ref).children.empty() ||
                other.node(other_it->second).children.empty()) {
                result._roots.emplace(value, merged);
            }
        }
    }
    result.coalesce_whole_access_chains();
    return result;
}

AccessTree::NodeRef AccessTree::_union_nodes(const AccessTree &a, NodeRef a_ref,
                                              const AccessTree &b, NodeRef b_ref) noexcept {
    auto &a_node = a.node(a_ref);
    auto &b_node = b.node(b_ref);
    if (a_node.children.empty()) {
        return _clone_node(a, a_ref);
    }
    if (b_node.children.empty()) {
        return _clone_node(b, b_ref);
    }
    // merge all children
    auto new_ref = add_node();
    luisa::unordered_set<AccessChainIndex, AccessChainIndexHash> all_indices;
    for (auto &[idx, _] : a_node.children) all_indices.emplace(idx);
    for (auto &[idx, _] : b_node.children) all_indices.emplace(idx);
    for (auto &idx : all_indices) {
        auto a_it = a_node.children.find(idx);
        auto b_it = b_node.children.find(idx);
        NodeRef child;
        if (a_it != a_node.children.end() && b_it != b_node.children.end()) {
            child = _union_nodes(a, a_it->second, b, b_it->second);
        } else if (a_it != a_node.children.end()) {
            child = _clone_node(a, a_it->second);
        } else {
            child = _clone_node(b, b_it->second);
        }
        node(new_ref).children.emplace(idx, child);
    }
    return new_ref;
}

AccessTree AccessTree::union_with(const AccessTree &other) const noexcept {
    AccessTree result;
    // common roots
    for (auto &[value, ref] : _roots) {
        auto other_it = other._roots.find(value);
        if (other_it != other._roots.end()) {
            auto merged = result._union_nodes(*this, ref, other, other_it->second);
            result._roots.emplace(value, merged);
        } else {
            auto cloned = result._clone_node(*this, ref);
            result._roots.emplace(value, cloned);
        }
    }
    // roots only in other
    for (auto &[value, ref] : other._roots) {
        if (!_roots.contains(value)) {
            auto cloned = result._clone_node(other, ref);
            result._roots.emplace(value, cloned);
        }
    }
    result.coalesce_whole_access_chains();
    return result;
}

static size_t type_member_count(const Type *type) noexcept {
    if (type->is_vector()) return type->dimension();
    if (type->is_matrix()) return type->dimension();
    if (type->is_array()) return type->dimension();
    if (type->is_structure()) return type->members().size();
    return 0u;
}

static const Type *type_member(const Type *type, size_t index) noexcept {
    if (type->is_vector()) return type->element();
    if (type->is_matrix()) return Type::vector(type->element(), type->dimension());
    if (type->is_array()) return type->element();
    if (type->is_structure()) return type->members()[index];
    return nullptr;
}

void AccessTree::_enumerate_all_children_for_subtraction(
    Value *value, const Type *type,
    luisa::vector<AccessChainIndex> &chain,
    const AccessTree &other,
    AccessTree &result) const noexcept {
    auto count = type_member_count(type);
    if (count == 0u) {
        // primitive/leaf type — check if contained in other
        if (!other.contains(value, chain)) {
            result.insert(value, chain);
        }
    } else {
        for (size_t i = 0u; i < count; i++) {
            chain.emplace_back(AccessChainIndex::make_static(static_cast<int32_t>(i)));
            _enumerate_all_children_for_subtraction(
                value, type_member(type, i), chain, other, result);
            chain.pop_back();
        }
    }
}

void AccessTree::_enumerate_for_subtraction(
    Value *value, const Type *type,
    const Node &access_node,
    luisa::vector<AccessChainIndex> &chain,
    const AccessTree &other,
    AccessTree &result) const noexcept {
    if (access_node.children.empty()) {
        // leaf → enumerate all children recursively
        _enumerate_all_children_for_subtraction(value, type, chain, other, result);
    } else {
        for (auto &[idx, child_ref] : access_node.children) {
            LUISA_ASSERT(idx.is_static(), "Dynamic access chains not supported in subtract.");
            chain.emplace_back(idx);
            auto child_type = type_member(type, static_cast<size_t>(idx.static_index));
            _enumerate_for_subtraction(value, child_type, node(child_ref), chain, other, result);
            chain.pop_back();
        }
    }
}

AccessTree AccessTree::subtract(const AccessTree &other) const noexcept {
    AccessTree result;
    luisa::vector<AccessChainIndex> chain;
    for (auto &[value, ref] : _roots) {
        chain.clear();
        auto type = value->type();
        _enumerate_for_subtraction(value, type, node(ref), chain, other, result);
    }
    result.coalesce_whole_access_chains();
    return result;
}

void AccessTree::_coalesce(NodeRef ref, const Type *type) noexcept {
    auto &n = node(ref);
    if (n.children.empty()) return;

    auto expected_count = type_member_count(type);
    size_t full_children = 0u;
    for (auto &[idx, child_ref] : n.children) {
        if (idx.is_static()) {
            auto child_type = type_member(type, static_cast<size_t>(idx.static_index));
            _coalesce(child_ref, child_type);
            if (node(child_ref).children.empty()) {
                full_children++;
            }
        }
    }
    if (expected_count > 0u && full_children == expected_count) {
        n.children.clear();// collapse to whole access
    }
}

void AccessTree::coalesce_whole_access_chains() noexcept {
    for (auto &[value, ref] : _roots) {
        _coalesce(ref, value->type());
    }
}

void AccessTree::_collapse_dynamic(NodeRef ref, const Type *type) noexcept {
    auto &n = node(ref);
    bool has_dynamic = std::any_of(
        n.children.begin(), n.children.end(),
        [](auto &p) { return p.first.is_dynamic(); });
    if (has_dynamic) {
        n.children.clear();
    } else {
        for (auto &[idx, child_ref] : n.children) {
            if (idx.is_static()) {
                auto child_type = type_member(type, static_cast<size_t>(idx.static_index));
                _collapse_dynamic(child_ref, child_type);
            }
        }
    }
}

void AccessTree::dynamic_access_chains_as_whole() noexcept {
    for (auto &[value, ref] : _roots) {
        _collapse_dynamic(ref, value->type());
    }
}

}// namespace luisa::compute::xir
