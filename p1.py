print("This program analyzes a list of numbers: original, unique, even/odd, primes.")

def is_prime(num):
    """Check if number is prime (handle negatives/1/0)."""
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True

try:
    n = int(input("Enter number of values (n): "))
    if n <= 0:
        raise ValueError("n must be positive")
    
    numbers = []
    print(f"Enter {n} numbers:")
    for i in range(n):
        num = int(input(f"Number {i+1}: "))
        numbers.append(num)
    
    print("\nOriginal list:", numbers)
    
    unique_numbers = list(set(numbers))
    print("Unique numbers:", unique_numbers)
    
    even_numbers = [num for num in numbers if num % 2 == 0]
    print("Even numbers:", even_numbers)
    
    odd_numbers = [num for num in numbers if num % 2 != 0]
    print("Odd numbers:", odd_numbers)
    
    prime_numbers = [num for num in numbers if is_prime(num)]
    print("Prime numbers:", prime_numbers)
    
    print("\nSummary:")
    print(f"Total: {len(numbers)}, Unique: {len(unique_numbers)}, Primes: {len(prime_numbers)}")
    
except ValueError as e:
    print(f"Error: {e}. Please enter integers only.")
