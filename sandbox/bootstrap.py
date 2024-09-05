import time

import numpy as np


def main() -> None:
    data = np.random.rand(5000)
    generator = np.random.default_rng(5)
    start_time = time.time()
    resamples = generator.choice(data, (10000, 5000), replace=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    start_time = time.time()
    resample_means = np.mean(resamples, axis=1)
    resample_means.var()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
