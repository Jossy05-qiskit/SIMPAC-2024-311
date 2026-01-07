#!/usr/bin/env python
"""
Simple runner pour tester le packing avec les parametres optimises.
"""
import sys
from env import MultiBinPackerEnv
from conveyor import FileConveyor
from agent import Agent
import tensorflow as tf
import os
import copy

# Config pallet
BIN_SIZE = (1200, 1200, 2000)
Z_GRIPPER_MAX = 1650
Z_GRIPPER_OFFSET = 745

def run_packing(method='rl', lookahead=32, filepath='./inputpack.txt', verbose=1):
    """Execute packing with optimized parameters."""
    
    print(f"\n[START] Packing with method={method}, lookahead={lookahead}")
    print(f"  Config: BIN={BIN_SIZE}, Z_MAX={Z_GRIPPER_MAX}mm")
    
    # Create conveyor
    conv = FileConveyor(filepath, k=lookahead)
    conv.reset()
    print(f"  Loaded {len(conv.items)} items from {filepath}")
    
    # Create environment WITH parameters
    env = MultiBinPackerEnv(
        n_bins=1,
        size=BIN_SIZE,
        bin_size=BIN_SIZE,
        z_gripper_max=Z_GRIPPER_MAX,
        z_gripper_offset=Z_GRIPPER_OFFSET,
        strict_no_xy_overlap=True,
        # Gap visuel XY : passer à 10mm pour éliminer l'entrelacement visible sur Delmia
        min_xy_gap=50
    )
    env.conveyor = conv
    print(f"  Environment created with strict_no_xy_overlap=True, min_xy_gap=50mm")
    
    # Load model
    model_path = f'./models/k={lookahead}.h5'
    if not os.path.exists(model_path):
        print(f"  ERROR: Model not found at {model_path}")
        return
    
    agent = Agent(env, visualize=True, train=False)
    agent.q_net = tf.keras.models.load_model(model_path, compile=False)
    agent.eps = 0.0
    print(f"  Model loaded: {agent.q_net.input_shape}")
    
    # Run packing
    print(f"\n[RUNNING] Placing items...")
    for episode_id, reward, utils in agent.run(n_episodes=1, verbose=verbose):
        print(f"  Episode {episode_id}: reward={reward:.4f}, items_placed={len(utils)}")
    
    # Check results
    if hasattr(env, 'final_packers') and env.final_packers:
        packers = env.final_packers
    elif hasattr(agent, 'final_packers') and agent.final_packers:
        packers = agent.final_packers
    else:
        print(f"  WARNING: No final_packers found")
        packers = env.used_packers
    
    total_items = sum(len(p.items) for p in packers)
    print(f"\n[RESULT] {total_items} items placed in {len(packers)} bin(s)")
    
    # Visualize
    from plot_all_bins import plot_all_bins
    plot_all_bins(packers, output_path='./outputs/bin_final.png')
    print(f"  Visualization saved to ./outputs/bin_final.png")
    
    return packers

if __name__ == '__main__':
    method = sys.argv[1] if len(sys.argv) > 1 else 'rl'
    lookahead = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    packers = run_packing(method=method, lookahead=lookahead, verbose=verbose)
