import numpy as np
from scipy.spatial.transform import Rotation


def sample(total_n, sample_n, fix_interval=True):
    ids = np.arange(total_n)
    if sample_n == total_n:
        return ids
    elif sample_n > total_n:
        return np.concatenate([
            ids, sample(total_n, sample_n - total_n, fix_interval)
        ])
    elif fix_interval:
        interval = total_n / sample_n
        output = []
        for i in range(sample_n):
            output.append(ids[int(interval * i)])
        return np.array(output)
    return np.random.choice(ids, sample_n, replace=False)


def xyzrpy_to_matrix(input):
    output = np.eye(4)
    output[:3, :3] = Rotation.from_euler("xyz", np.array(input[3:])).as_matrix()
    output[:3, 3] = input[:3]
    return output
