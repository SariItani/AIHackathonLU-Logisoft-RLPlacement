import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random
from logisoft_hackathon_rl_optimizer import VehicleYardEnvironment, DQNAgent

# Configuration
st.set_page_config(layout="wide")
GRID_WIDTH, GRID_HEIGHT = 20, 15

# Load model
@st.cache_resource
def load_agent(model_path):
    agent = DQNAgent(state_size=1000, action_size=300)
    agent.load_model(model_path)
    return agent

# Initialize environment and state
if "env" not in st.session_state:
    st.session_state.env = VehicleYardEnvironment()
    st.session_state.state = st.session_state.env.reset()
    st.session_state.agent = None
    st.session_state.total_reward = 0
    st.session_state.done = False

# Sidebar controls
st.sidebar.title("Controls")

# Load trained model
model_file = st.sidebar.file_uploader("Upload Trained Model (.pth)", type="pth")
if model_file:
    with open("uploaded_model.pth", "wb") as f:
        f.write(model_file.getbuffer())
    st.session_state.agent = load_agent("uploaded_model.pth")
    st.sidebar.success("Model loaded!")

# Reset simulation
if st.sidebar.button("Reset Simulation"):
    st.session_state.env = VehicleYardEnvironment()
    st.session_state.state = st.session_state.env.reset()
    st.session_state.total_reward = 0
    st.session_state.done = False

# Add vehicle to queue
st.sidebar.markdown("### Queue Controls")
vehicle_type = st.sidebar.selectbox("Add Vehicle Type", ["car", "suv", "truck", "bus", "van"])
if st.sidebar.button("Add Vehicle"):
    generator = st.session_state.env.generator if hasattr(st.session_state.env, 'generator') else None
    if generator is None:
        from logisoft_hackathon_rl_optimizer import VehicleDatasetGenerator
        generator = VehicleDatasetGenerator(st.session_state.env.width, st.session_state.env.height)
        st.session_state.env.generator = generator
    vehicle = generator.generate_scenario(1)[0]
    vehicle['type'] = vehicle_type
    st.session_state.env.vehicle_queue.append(vehicle)

    # If no current vehicle is active, load the next one
    if st.session_state.env.current_vehicle is None:
        st.session_state.env.current_vehicle = st.session_state.env.vehicle_queue.pop(0)

    # Refresh state
    st.session_state.state = st.session_state.env._get_state()

# Clear the queue
if st.sidebar.button("Clear Queue"):
    st.session_state.env.vehicle_queue.clear()
    st.session_state.env.current_vehicle = None
    st.session_state.state['queue_length'] = 0
    st.session_state.done = False

# Step simulation
def run_step():
    if st.session_state.done:
        return

    env = st.session_state.env
    state = st.session_state.state
    agent = st.session_state.agent

    valid_actions = env.get_valid_actions()
    action = 'skip'

    if agent and valid_actions:
        action = agent.act(state, valid_actions + ['skip'])
    elif valid_actions:
        action = random.choice(valid_actions)

    next_state, reward, done, info = env.step(action)
    st.session_state.state = next_state
    st.session_state.total_reward += reward
    st.session_state.done = done

st.sidebar.button("Step", on_click=run_step)

# Visualization
def draw_yard(state):
    fig, ax = plt.subplots(figsize=(10, 6))
    grid = state['grid']
    zone_map = state['zone_map']

    # Draw roads and parking zones
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if zone_map[y, x] == 0:
                rect = patches.Rectangle((x, y), 1, 1, color='lightgray', alpha=0.3)
            else:
                rect = patches.Rectangle((x, y), 1, 1, edgecolor='white', facecolor='none', linewidth=0.2)
            ax.add_patch(rect)

    # Draw placed vehicles
    colors = plt.cm.Set3(np.linspace(0, 1, len(st.session_state.env.placed_vehicles)))
    for i, vehicle in enumerate(st.session_state.env.placed_vehicles):
        rect = patches.Rectangle(
            (vehicle['x'], vehicle['y']),
            vehicle['width'], vehicle['height'],
            linewidth=1, edgecolor='black',
            facecolor=colors[i], alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            vehicle['x'] + vehicle['width']/2,
            vehicle['y'] + vehicle['height']/2,
            vehicle['type'].upper(),
            ha='center', va='center',
            fontsize=6, fontweight='bold'
        )

    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Vehicle Yard State')
    st.pyplot(fig)

# Display yard and stats
st.title("🚛 RL Vehicle Placement Simulator")
draw_yard(st.session_state.state)

with st.expander("Simulation Metrics"):
    st.write(f"**Vehicles Placed:** {len(st.session_state.env.placed_vehicles)}")
    st.write(f"**Occupancy Rate:** {st.session_state.state['occupancy_rate']:.2%}")
    st.write(f"**Queue Length:** {st.session_state.state['queue_length']}")
    st.write(f"**Steps:** {st.session_state.state['step_count']}")
    st.write(f"**Total Reward:** {st.session_state.total_reward}")

with st.expander("📦 Next Vehicles in Queue"):
    queue_preview = st.session_state.env.vehicle_queue[:5]
    if not queue_preview:
        st.write("Queue is empty.")
    else:
        for i, v in enumerate(queue_preview):
            st.write(f"{i+1}. **{v['type'].upper()}** – {v['width']}x{v['height']}, Duration: {v['duration']}, Priority: {v['priority']}")
