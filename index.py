import numpy as np
from collections import Counter

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f']

class Node:
    def __init__(self, symbol=None, weight=0, parent=None):
        self.symbol = symbol
        self.weight = weight
        self.parent = parent
        self.left = None
        self.right = None

class AdaptiveHuffmanTree:
    def __init__(self):
        self.root = None
        self.NYT = Node(symbol="NYT", weight=0)
        self.nodes = {'NYT': self.NYT}
        self.symbols_to_nodes = {}

    def update_tree(self, symbol):
        if symbol not in self.symbols_to_nodes:
            self.add_new_symbol(symbol)
        else:
            self.increment_weight(self.symbols_to_nodes[symbol])
        self.rebalance_tree()

    def add_new_symbol(self, symbol):
        new_nyt = Node(symbol="NYT", weight=0)
        new_symbol_node = Node(symbol=symbol, weight=1)
        nyt_parent = self.NYT.parent

        internal_node = Node(weight=1)
        internal_node.left = new_nyt
        internal_node.right = new_symbol_node
        new_nyt.parent = internal_node
        new_symbol_node.parent = internal_node

        if nyt_parent:
            internal_node.parent = nyt_parent
            if nyt_parent.left == self.NYT:
                nyt_parent.left = internal_node
            else:
                nyt_parent.right = internal_node
        else:
            self.root = internal_node

        self.NYT = new_nyt
        self.nodes[new_nyt.symbol] = new_nyt
        self.nodes[new_symbol_node.symbol] = new_symbol_node
        self.symbols_to_nodes[symbol] = new_symbol_node
        self.increment_weight(new_symbol_node.parent)

    def increment_weight(self, node):
        while node:
            node.weight += 1
            node = node.parent

    def rebalance_tree(self):
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda x: (x.weight, id(x)))
        if nodes[-1] != self.root:
            self.rebuild_tree(nodes)

    def rebuild_tree(self, nodes):
        new_root = nodes[-1]
        if new_root.parent:
            parent = new_root.parent
            if parent.left == new_root:
                parent.left = None
            else:
                parent.right = None

        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            new_internal = Node(weight=left.weight + right.weight)
            new_internal.left = left
            new_internal.right = right
            left.parent = new_internal
            right.parent = new_internal
            nodes.append(new_internal)
            nodes.sort(key=lambda x: (x.weight, id(x)))

        self.root = nodes[0]

    def encode_symbol(self, symbol):
        code = ""
        if symbol in self.symbols_to_nodes:
            current = self.symbols_to_nodes[symbol]
        else:
            current = self.NYT

        while current and current.parent:
            if current.parent.left == current:
                code = "0" + code
            else:
                code = "1" + code
            current = current.parent

        return code

    def adjust_probabilities(self, symbol_frequencies):
        total_weight = sum(symbol_frequencies.values())
        
        for symbol, frequency in symbol_frequencies.items():
            if symbol in self.symbols_to_nodes:
                node = self.symbols_to_nodes[symbol]
                node.weight = frequency / total_weight
                self.rebalance_tree()

    def encode(self, data):
        encoded_data = ""
        for symbol in data:
            encoded_data += self.encode_symbol(symbol)
            self.update_tree(symbol)
        return encoded_data

    def decode(self, encoded_data):
        current = self.root
        decoded_output = ""
        i = 0
        while i < len(encoded_data):
            if current.left is None and current.right is None:
                decoded_output += current.symbol
                self.update_tree(current.symbol)
                current = self.root
            else:
                if encoded_data[i] == '0':
                    current = current.left
                else:
                    current = current.right
                i += 1

        return decoded_output


def adapt_probabilities(sequence, window_size=100):
    frequencies = Counter(sequence[:window_size])
    total = sum(frequencies.values())
    probs = {symbol: frequencies[symbol] / total for symbol in ALPHABET}
    return probs


def generate_sequence(probs, length):
    symbols = ALPHABET
    sequence = np.random.choice(symbols, length, p=probs)
    return ''.join(sequence)

def compression_ratio(original_length, compressed_length):
    return original_length / compressed_length


distribution = [
    [0.05, 0.1, 0.15, 0.18, 0.22, 0.3],
    [0.15, 0.1, 0.05, 0.3, 0.22, 0.18]
]


fixed_sequence = []
changing_sequence = []

for i in range(10):
    dist_idx = i % 2
    changing_sequence.extend(generate_sequence(distribution[dist_idx], 200))
    fixed_sequence.extend(generate_sequence(distribution[0], 200))

changing_sequence = ''.join(changing_sequence)
fixed_sequence = ''.join(fixed_sequence)

tree_changing = AdaptiveHuffmanTree()
tree_fixed = AdaptiveHuffmanTree()

encoded_sequence_fixed = ""
encoded_sequence_changing = ""

window_size = 100

for i in range(0, 2000, window_size):
    window_changing = changing_sequence[:i+window_size] if i+window_size <= len(changing_sequence) else changing_sequence
    window_fixed = fixed_sequence[:i+window_size] if i+window_size <= len(fixed_sequence) else fixed_sequence
    
    adapt_probs_changing = adapt_probabilities(window_changing, window_size)
    adapt_probs_fixed = adapt_probabilities(window_fixed, window_size)

    tree_changing.adjust_probabilities(adapt_probs_changing)
    
    encoded_sequence_changing += tree_changing.encode(changing_sequence[i:i+window_size])
    encoded_sequence_fixed += tree_fixed.encode(fixed_sequence[i:i+window_size])

original_size_fixed = len(fixed_sequence) * np.log2(len(ALPHABET))
fixed_compression_ratio = compression_ratio(original_size_fixed, len(encoded_sequence_fixed))


original_size_changing = len(changing_sequence) * np.log2(len(ALPHABET))
changing_compression_ratio = compression_ratio(original_size_changing, len(encoded_sequence_changing))

print(f"{changing_compression_ratio}")
