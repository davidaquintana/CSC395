
import numpy as np
import string
import random


#Part 1
def build_transition_matrix(file_path):
    alphabet = string.ascii_lowercase + string.ascii_uppercase + ' ,.' # all characters
    alphabet_index = {char: index for index, char in enumerate(alphabet)}
    transition_matrix = np.full((len(alphabet), len(alphabet)), fill_value=1e-20)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    prev_char = None
    for char in text:
        if char in alphabet_index:
            if prev_char is not None:
                i = alphabet_index[prev_char]
                j = alphabet_index[char]
                transition_matrix[i][j] += 1
            prev_char = char
    
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix, alphabet

def random_reverse_cipher(alphabet):
    cipher_list = list(alphabet)
    random.shuffle(cipher_list)
    return ''.join(cipher_list)

def encipher(message, cipher, alphabet):
    cipher_index = {char: cipher[i] for i, char in enumerate(alphabet)}
    return ''.join(cipher_index.get(char, char) for char in message)

def new_reverse_cipher(cipher):
    cipher_list = list(cipher)
    a, b = random.sample(range(len(cipher_list)), 2)
    cipher_list[a], cipher_list[b] = cipher_list[b], cipher_list[a]
    return ''.join(cipher_list)

def acceptance_probability(old_msg, new_msg, transition_matrix, alphabet):
    alphabet_index = {char: index for index, char in enumerate(alphabet)}
    log_prob_old = log_prob_new = 0
    
    for i in range(len(old_msg) - 1):
        if old_msg[i] in alphabet_index and old_msg[i + 1] in alphabet_index:
            log_prob_old += np.log(transition_matrix[alphabet_index[old_msg[i]], alphabet_index[old_msg[i + 1]]])
        if new_msg[i] in alphabet_index and new_msg[i + 1] in alphabet_index:
            log_prob_new += np.log(transition_matrix[alphabet_index[new_msg[i]], alphabet_index[new_msg[i + 1]]])
    
    return min(np.exp(log_prob_new - log_prob_old), 1)

def metropolis_hastings(scrambled_msg, transition_matrix, alphabet, iterations=10000):
    current_cipher = random_reverse_cipher(alphabet)
    current_msg = encipher(scrambled_msg, current_cipher, alphabet)
    best_cipher, best_msg = current_cipher, current_msg
    best_prob = -np.inf
    
    for _ in range(iterations):
        proposed_cipher = new_reverse_cipher(current_cipher)
        proposed_msg = encipher(scrambled_msg, proposed_cipher, alphabet)
        a_prob = acceptance_probability(current_msg, proposed_msg, transition_matrix, alphabet)
        
        if a_prob > best_prob:
            best_cipher, best_msg, best_prob = proposed_cipher, proposed_msg, a_prob
        
        if random.random() < a_prob:
            current_cipher, current_msg = proposed_cipher, proposed_msg
    
    return best_cipher, best_msg



# Part 2
# Security enhancement methods
def insert_random_characters(message, interval=5, seed=42):
    random.seed(seed)
    altered_message = ""
    alphabet = string.ascii_letters + ",. "
    for i, char in enumerate(message):
        altered_message += char
        if (i + 1) % interval == 0:
            altered_message += random.choice(alphabet)
    return altered_message

def shuffle_words(message, seed=42):
    random.seed(seed)
    words = message.split()
    random.shuffle(words)
    return ' '.join(words)

def reverse_insert_random_characters(message, interval=5, seed=42):
    random.seed(seed)
    return ''.join([char for i, char in enumerate(message) if (i + 1) % (interval + 1) != 0])

def unshuffle_words(shuffled_message, seed=42):
    random.seed(seed)
    words = shuffled_message.split()
    indices = list(range(len(words)))
    random.shuffle(indices)
    original_order = [0] * len(words)
    for original_index, shuffled_index in enumerate(indices):
        original_order[shuffled_index] = words[original_index]
    return ' '.join(original_order)

if __name__ == "__main__":
    transition_matrix, alphabet = build_transition_matrix('c:\Users\david\OneDrive\Desktop\Misc\School\Code\School\CSC395\hw4\WarAndPeace.txt')
    scrambled_message = "Your scrambled message here"
    best_cipher, decrypted_message = metropolis_hastings(scrambled_message, transition_matrix, alphabet)
    print(f"Best Cipher: {best_cipher}\nDecrypted Message: {decrypted_message}")
    
    original_message = "This is a test message for the Metropolis-Hastings algorithm."

    # Apply security enhancements
    seed = 42
    altered_message = insert_random_characters(original_message, seed=seed)
    altered_shuffled_message = shuffle_words(altered_message, seed=seed)
    
    # For demonstration, scramble using a random reverse cipher (this should be known or the same for decryption)
    initial_reverse_cipher = random_reverse_cipher(alphabet)
    scrambled_message = encipher(altered_shuffled_message, initial_reverse_cipher, alphabet)

    # Attempt to decipher the scrambled message
    best_reverse_cipher, _ = metropolis_hastings(scrambled_message, transition_matrix, alphabet)
    deciphered_message = encipher(scrambled_message, best_reverse_cipher, alphabet)

    # Reverse security enhancements to recover the original message
    message_after_unshuffling = unshuffle_words(deciphered_message, seed=seed)
    original_recovered = reverse_insert_random_characters(message_after_unshuffling, interval=5, seed=seed)

    print("Original Message:", original_message)
    print("Altered & Shuffled Message:", altered_shuffled_message)
    print("Scrambled Message:", scrambled_message)
    print("Recovered Original Message:", original_recovered)
