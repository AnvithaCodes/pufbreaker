"""
Basic tests for PUFBreaker components.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pufbreaker import ArbiterPUF, XORPUF, LRAttack, MLPAttack


def test_arbiter_puf():
    """Test basic Arbiter PUF functionality."""
    print("Testing Arbiter PUF...")
    
    puf = ArbiterPUF(n_stages=64, noise=0.01, seed=42)
    
    # Test single challenge
    challenge = np.array([1, 0, 1, 0] * 16)
    response = puf.evaluate(challenge)
    assert response in [0, 1], "Response should be binary"
    
    # Test batch generation
    X, y = puf.generate_dataset(100)
    assert X.shape == (100, 64), f"Wrong challenge shape: {X.shape}"
    assert y.shape == (100,), f"Wrong response shape: {y.shape}"
    assert np.all((y == 0) | (y == 1)), "Responses should be binary"
    
    print("✓ Arbiter PUF tests passed")


def test_xor_puf():
    """Test XOR PUF functionality."""
    print("Testing XOR PUF...")
    
    xor_puf = XORPUF(n_stages=64, k=2, noise=0.01, seed=42)
    
    # Test dataset generation
    X, y = xor_puf.generate_dataset(100)
    assert X.shape == (100, 64), f"Wrong challenge shape: {X.shape}"
    assert y.shape == (100,), f"Wrong response shape: {y.shape}"
    assert np.all((y == 0) | (y == 1)), "Responses should be binary"
    
    # Test that XOR is different from single arbiter
    arbiter = ArbiterPUF(n_stages=64, noise=0.01, seed=42)
    X_test, _ = arbiter.generate_dataset(100, seed=999)
    y_xor = xor_puf.evaluate_batch(X_test)
    y_arb = arbiter.evaluate_batch(X_test)
    
    # XOR output should differ from single arbiter
    diff_rate = np.mean(y_xor != y_arb)
    assert diff_rate > 0.1, f"XOR should differ from arbiter (diff rate: {diff_rate})"
    
    print("✓ XOR PUF tests passed")


def test_lr_attack():
    """Test Logistic Regression attack."""
    print("Testing LR Attack on Arbiter PUF...")
    
    # Create Arbiter PUF
    puf = ArbiterPUF(n_stages=64, noise=0.01, seed=42)
    
    # Generate training data
    X_train, y_train = puf.generate_dataset(1000, seed=100)
    X_test, y_test = puf.generate_dataset(500, seed=200)
    
    # Train attack
    attack = LRAttack(C=1.0)
    attack.fit(X_train, y_train)
    
    # Test accuracy
    train_acc = attack.score(X_train, y_train)
    test_acc = attack.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {attack.training_time:.2f}s")
    
    # LR should work well on Arbiter PUF
    assert test_acc > 0.85, f"LR attack too weak on Arbiter: {test_acc}"
    
    print("✓ LR Attack tests passed")


def test_lr_fails_on_xor():
    """Test that LR fails on XOR PUF."""
    print("Testing LR Attack failure on XOR PUF...")
    
    # Create XOR PUF
    xor_puf = XORPUF(n_stages=64, k=2, noise=0.01, seed=42)
    
    # Generate data
    X_train, y_train = xor_puf.generate_dataset(2000, seed=100)
    X_test, y_test = xor_puf.generate_dataset(500, seed=200)
    
    # Train attack
    attack = LRAttack(C=1.0)
    attack.fit(X_train, y_train)
    
    test_acc = attack.score(X_test, y_test)
    
    print(f"  Test accuracy on XOR: {test_acc:.4f}")
    
    # LR should fail on XOR (barely better than random guessing)
    assert test_acc < 0.70, f"LR should fail on XOR, got {test_acc}"
    
    print("✓ LR correctly fails on XOR PUF")


def test_mlp_attack():
    """Test Neural Network attack."""
    print("Testing MLP Attack on XOR PUF...")
    
    # Create XOR PUF
    xor_puf = XORPUF(n_stages=64, k=2, noise=0.01, seed=42)
    
    # Generate data (more samples for NN)
    X_train, y_train = xor_puf.generate_dataset(3000, seed=100)
    X_test, y_test = xor_puf.generate_dataset(500, seed=200)
    
    # Train attack
    attack = MLPAttack(hidden_layers=(64, 32), max_iter=200)
    attack.fit(X_train, y_train, verbose=False)
    
    # Test accuracy
    train_acc = attack.score(X_train, y_train)
    test_acc = attack.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Training time: {attack.training_time:.2f}s")
    
    # MLP should work on XOR PUF
    assert test_acc > 0.75, f"MLP attack too weak on XOR: {test_acc}"
    
    print("✓ MLP Attack tests passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running PUFBreaker Tests")
    print("="*60)
    print()
    
    try:
        test_arbiter_puf()
        print()
        test_xor_puf()
        print()
        test_lr_attack()
        print()
        test_lr_fails_on_xor()
        print()
        test_mlp_attack()
        print()
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print()
        print("="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60)
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()