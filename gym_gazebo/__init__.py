import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------



register(
    id='GazeboCustomSnakeMonsterDDPG-v0',
    entry_point='gym_gazebo.envs:GazeboCustomSnakeMonsterDDPG',
    # More arguments here
)

