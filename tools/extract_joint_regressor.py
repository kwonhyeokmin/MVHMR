import torch
import constants
import numpy as np


mlkit = {
    'nose':		    332,
    'reye':		    6260,
    'leye':		    2800,
    'rear':		    4071,
    'lear':		    583,
    'l_mcp_joint_2':		2135,
    'l_mcp_joint_5':		2193,
    'r_mcp_joint_2':		5595,
    'r_mcp_joint_5':		5525,
    'l_big_toe':		3337,
    'l_little_toe':		3344,
    'r_big_toe': 6739,
    'r_little_toe': 6745,
}

# Load joint regressor
J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_ORI))

extra = torch.zeros((len(mlkit.keys()), 6890))
for i, (k,v) in enumerate(mlkit.items()):
    extra[i, v] = 1
J_regressor = torch.vstack((J_regressor, extra))

mlkit_regressor = torch.zeros((29, 6890))
mlkit_regressor[0, :] = J_regressor[0, :]
mlkit_regressor[1, :] = J_regressor[1, :]
mlkit_regressor[2, :] = J_regressor[4, :]
mlkit_regressor[3, :] = J_regressor[7, :]
mlkit_regressor[4, :] = J_regressor[33, :]
mlkit_regressor[5, :] = J_regressor[34, :]
mlkit_regressor[6, :] = J_regressor[2, :]
mlkit_regressor[7, :] = J_regressor[5, :]
mlkit_regressor[8, :] = J_regressor[8, :]
mlkit_regressor[9, :] = J_regressor[35, :]
mlkit_regressor[10, :] = J_regressor[36, :]
mlkit_regressor[11, :] = J_regressor[3, :]
mlkit_regressor[12, :] = J_regressor[9, :]
mlkit_regressor[13, :] = J_regressor[12, :]
mlkit_regressor[14, :] = J_regressor[16, :]
mlkit_regressor[15, :] = J_regressor[18, :]
mlkit_regressor[16, :] = J_regressor[20, :]
mlkit_regressor[17, :] = J_regressor[29, :]
mlkit_regressor[18, :] = J_regressor[30, :]
mlkit_regressor[19, :] = J_regressor[17, :]
mlkit_regressor[20, :] = J_regressor[19, :]
mlkit_regressor[21, :] = J_regressor[21, :]
mlkit_regressor[22, :] = J_regressor[31, :]
mlkit_regressor[23, :] = J_regressor[32, :]
mlkit_regressor[24, :] = J_regressor[24, :]
mlkit_regressor[25, :] = J_regressor[26, :]
mlkit_regressor[26, :] = J_regressor[28, :]
mlkit_regressor[27, :] = J_regressor[25, :]
mlkit_regressor[28, :] = J_regressor[27, :]

np.save('data/J_regressor_mlkit.npy', mlkit_regressor.numpy())
