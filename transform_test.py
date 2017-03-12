import pickle
import matplotlib.pyplot as plt
from numpy import *
from full_pipeline import *
import scipy.spatial.distance
import operator


def points_in_circle(r, n=100):
    return [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r) for x in xrange(0, n+1)]

enable_translation = True
enable_scaling = True
enable_rotation = True

# Test variable ranges; all ranges are proportions of space size [0, 1]
noise_range = linspace(0, 0.5, 10)
if enable_translation:
    translation_range = map(lambda x: points_in_circle(x, 10), linspace(0.05, 0.5, 21))
else:
    translation_range = [[(0, 0)]]  # Identity
if enable_scaling:
    scaling_range = linspace(-0.5, 0.5, 20)
else:
    scaling_range = [1]  # Identity
if enable_rotation:
    rotation_range = linspace(-pi, pi, 20)
else:
    rotation_range = [0]  # Identity
num_point_range = linspace(5, 5, 1)  # Identity
iterations = 50

# Test translation only
test_combinations = list(itertools.product(*[num_point_range, noise_range, translation_range, scaling_range, rotation_range]))
translations = []
scalings = []
points = []
rotations = []
noises = []
errors = []
for idx, (n, noise_level, t, s, r) in enumerate(test_combinations):
    if idx % 100 == 0:
        print('Running test combination {0}/{1}.'.format(idx, len(test_combinations)))
    results = []
    expected_magnitude = linalg.norm([t[0][0], t[0][1]])
    co, si = cos(r), sin(r)
    rot_matrix = array([[co, -si], [si, co]])
    for i in range(0, iterations):
        # Generate noised test points
        correct, input = generate_random_test_points(number_of_points=int(n), dimension=2, shuffle_points=False,
                                                     noise=noise_level, num_rerandomed_points=0)
        for x, y in t:
            transformed_input = [list((array(new_point) + array([x, y])).dot(rot_matrix)*s) for new_point in
                     input]  # Apply known translation
            # noinspection PyRedeclaration
            rot, scaling, translation = similarity_transform(array(correct), array(transformed_input))

            output = [list((array(new_point) - array(translation)).dot(rot) / scaling) for new_point in
                      transformed_input]  # Apply known translation

            error = mean(map(lambda x: scipy.spatial.distance.euclidean(x[0], x[1]), zip(input, output)))
            results.append(error)

    scalings.append(s)
    rotations.append(r)
    points.append(n)
    errors.append(mean(results))
    translations.append(expected_magnitude)
    noises.append(noise_level)


def plot_heatmap(x_d, y_d, e, x_r, y_r, x_axis_label, fig_num):
    vis_shape = (len(y_r), len(x_r))
    min_error = min(e)
    max_error = max(e)
    print("shape={0}".format(vis_shape))
    print("min={0}, max={1}".format(min_error, max_error))
    x = reshape(array(x_d), vis_shape)
    y = reshape(array(y_d), vis_shape)
    z = reshape(array(e), vis_shape)
    plt.figure(fig_num)
    plt.contourf(x, y, z, cmap='RdBu', vmin=min_error, vmax=max_error)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.xlabel(x_axis_label)
    plt.ylabel("Noise Level (proportion)")

if enable_translation:
    all_data = zip(points, noises, errors, translations, scalings, rotations)
    all_data_sorted = sorted(all_data, key=operator.itemgetter(3))
    _, ns, es, ts, _, _ = zip(*all_data_sorted)
    partition_length = len(all_data)/(len(translation_range)*len(noise_range))
    ns = mean(array(ns).reshape(-1, partition_length), axis=1)
    es = mean(array(es).reshape(-1, partition_length), axis=1)
    ts = mean(array(ts).reshape(-1, partition_length), axis=1)
    plot_heatmap(ts, ns, es, translation_range, noise_range, "Translation Magnitude (proportion)", 1)
if enable_scaling:
    all_data = zip(points, noises, errors, translations, scalings, rotations)
    all_data_sorted = sorted(all_data, key=operator.itemgetter(4))
    _, ns, es, _, ss, _ = zip(*all_data_sorted)
    partition_length = len(all_data) / (len(scaling_range) * len(noise_range))
    ns = mean(array(ns).reshape(-1, partition_length), axis=1)
    es = mean(array(es).reshape(-1, partition_length), axis=1)
    ss = mean(array(ss).reshape(-1, partition_length), axis=1)
    plot_heatmap(ss, ns, es, scaling_range, noise_range, "Scaling Magnitude (proportion)", 2)
if enable_rotation:
    all_data = zip(points, noises, errors, translations, scalings, rotations)
    all_data_sorted = sorted(all_data, key=operator.itemgetter(5))
    _, ns, es, _, _, rs = zip(*all_data_sorted)
    partition_length = len(all_data) / (len(rotation_range) * len(noise_range))
    ns = mean(array(ns).reshape(-1, partition_length), axis=1)
    es = mean(array(es).reshape(-1, partition_length), axis=1)
    rs = mean(array(rs).reshape(-1, partition_length), axis=1)
    plot_heatmap(rs, ns, es, rotation_range, noise_range, "Rotation Magnitude (radians)", 3)

plt.show()
