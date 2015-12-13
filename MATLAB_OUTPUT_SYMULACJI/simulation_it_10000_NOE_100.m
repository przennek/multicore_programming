% ---------------------------------------------------------- %
allocation_time = 0.11632;
cpu_compution_time = 52.4221;
gpu_compution_time = 273840;
copyback_time = 0;
% ---------------------------------------------------------- %
number_of_particles = 100;
number_of_iterations = 10000;
% ---------------------------------------------------------- %
displacement = [58.5034, 38.0069, 37.6129, 101.333, 603.543, 438.096, 91.4653, 117.323, 235.502, 337.011, 221.468, 44.437, 411.051, 141.03, 78.4115, 149.055, 601.9, 65.6429, 496.781, 7.62201, 365.523, 1081.78, 200.247, 317.861, 173.328, 829.997, 1093.32, 140.481, 150.576, 274.503, 399.691, 699.757, 805.959, 35.866, 103.037, 103.835, 554.492, 552.796, 276.487, 0.507634, 100.171, 400.161, 64.1361, 1267.4, 121.497, 343.259, 685.013, 275.111, 586.831, 403.503, 124.967, 380.62, 54.0387, 797.143, 47.7027, 416.611, 850.231, 196.403, 129.56, 335.699, 779.676, 283.559, 331.011, 58.9045, 422.066, 373.881, 725.32, 608.975, 20.7304, 391.84, 515.327, 1077.03, 851.642, 5.95418, 170.777, 167.553, 315.266, 148.546, 53.4718, 145.637, 36.258, 476.28, 139.134, 306.67, 100.711, 325.674, 137.283, 32.3049, 164.477, 393.334, 115.342, 33.5167, 221.096, 69.8525, 204.721, 384.669, 35.8275, 1740.41, 72.3046, 349.828];
final_positions = [[21.9526, -54.2285], [32.2634, -20.0897], [13.3136, 35.1779], [97.2697, -28.4081], [495.269, 344.925], [-418.96, -128.065], [-70.2827, -58.5342], [114.283, 26.533], [39.0366, -232.244], [77.3484, 328.015], [-210.553, -68.6693], [42.8043, -11.9347], [-323.789, 253.227], [86.0773, -111.715], [-19.6814, -75.9013], [135.949, -61.1179], [500.252, 334.712], [3.95663, 65.5235], [478.747, 132.639], [-0.563411, -7.60116], [-350.669, 103.143], [-160.735, 1069.78], [-196.8, -36.9966], [-20.9888, 317.168], [-29.5464, -170.791], [-365.975, 744.955], [931.244, -572.822], [-39.1637, 134.912], [138.523, 59.0292], [-206.433, 180.934], [371.279, 148.002], [527.651, -459.613], [681.554, 430.18], [-34.5724, 9.54585], [-102.109, -13.794], [-2.86645, -103.795], [196.187, -518.626], [532.725, 147.604], [229.475, 154.229], [0.132987, 0.489905], [-13.9039, -99.2019], [348.214, 197.17], [48.5836, 41.8697], [432.223, -1191.42], [120.558, 15.0712], [237.234, 248.086], [-418.747, 542.12], [-273.288, -31.6202], [253.355, -529.322], [402.292, 31.2482], [-123.979, -15.6789], [34.1015, -379.089], [44.7356, 30.3135], [-524.906, 599.926], [45.9823, 12.6955], [-277.35, -310.872], [-191.305, 828.429], [-2.03517, -196.392], [72.8908, 107.111], [-122.594, 312.513], [670.639, 397.665], [152.386, 239.132], [-230.191, 237.866], [35.3062, -47.1509], [-370.813, -201.587], [340.735, 153.903], [-603.718, 402.013], [561.829, 234.943], [-19.1639, -7.90536], [-369.581, -130.188], [452.063, 247.388], [829.108, 687.442], [850.927, -34.8771], [-4.05319, 4.36164], [-65.0243, 157.914], [-25.431, -165.611], [-315.265, -0.750516], [-114.092, -95.1264], [33.5279, 41.6547], [133.199, 58.8896], [-31.2735, 18.347], [-391.753, -270.874], [-53.8293, -128.3], [-288.667, 103.525], [-21.9427, -98.2912], [255.37, -202.113], [134.502, 27.4939], [-24.2996, 21.287], [160.171, -37.3887], [-117.981, -375.223], [89.2536, 73.0581], [-20.8621, 26.2325], [-17.806, -220.378], [44.7188, 53.6619], [204.161, -15.1336], [200.207, -328.462], [5.61557, -35.3847], [1429.62, 992.588], [20.2805, -69.4021], [241.003, 253.569]];
total_distances = [100.911, 194.463, 184.442, 85.4981, 261.247, 242.817, 248.22, 76.4289, 114.214, 210.391, 208.951, 180.284, 164.99, 130.117, 183.316, 136.379, 314.336, 150.817, 210.628, 215.455, 202.748, 227.016, 181.643, 123.809, 218.418, 255.708, 312.902, 254.884, 153.593, 256.339, 211.042, 254.266, 282.041, 131.366, 132.089, 108.188, 200.213, 213.52, 132.734, 232.799, 228.569, 286.388, 96.9511, 278.238, 188.345, 264.959, 244.842, 222.509, 285.647, 212.389, 78.3841, 193.329, 228.539, 266.564, 172.547, 241.626, 275.084, 166.397, 112.414, 110.671, 305.913, 298.417, 234.664, 113.401, 228.685, 155.585, 297.903, 316.798, 233.347, 246.306, 220.105, 316.614, 262.035, 105.614, 223.305, 192.93, 193.649, 227.977, 169.353, 139.456, 153.139, 230.101, 207.029, 174.955, 135.812, 374.122, 162.435, 273.309, 131.666, 260.521, 92.0816, 129.257, 164.966, 167.268, 257.705, 249.626, 130.779, 318.539, 154.886, 282.536];
final_velocities = [1.8524, 3.19332, 3.0602, 1.1293, 2.62774, 6.59997, 4.62303, 0.698492, 4.33096, 5.83672, 0.980156, 4.19225, 1.56982, 0.881963, 3.24964, 2.12953, 5.42132, 2.68631, 4.18057, 3.88779, 2.38251, 1.19393, 5.27396, 0.212322, 3.03524, 3.33451, 4.41645, 5.1272, 1.51838, 6.04434, 2.60507, 3.03562, 4.484, 2.68619, 3.38297, 1.62144, 1.87355, 2.97322, 1.49358, 3.99406, 4.36144, 4.31564, 1.75777, 3.86265, 3.50357, 4.81125, 1.39827, 6.05205, 6.80868, 4.37643, 0.238585, 5.02798, 3.97158, 3.19375, 2.46917, 1.97324, 3.38107, 4.00881, 0.189265, 0.338458, 4.64382, 5.1055, 4.51246, 1.96114, 5.85322, 1.52986, 4.40496, 7.77849, 4.19887, 4.59494, 0.697529, 5.11175, 6.10245, 2.47163, 4.66393, 4.04637, 4.92603, 4.74008, 2.57964, 1.11583, 3.10127, 5.6095, 2.76697, 0.660522, 1.10207, 8.43824, 1.66907, 4.67603, 0.999177, 5.46645, 0.694765, 2.20168, 2.95161, 2.5179, 5.97086, 3.03835, 2.16824, 3.93666, 3.16173, 4.19636];