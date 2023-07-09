#%%
# standard imports
import urllib.request  # to download MNIST
import gzip            # to download MNIST
from time import time

# Numerics
import jax
import jaxlib
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt
import numpy as np
jax.config.update("jax_enable_x64", True)  # use double-precision numbers
jax.config.update("jax_platform_name", "cpu")  # we don't need GPU here

# Plotting
from matplotlib import pyplot as plt
from tueplots import bundles

plt.rcParams.update({"figure.dpi": 200})
plt.rcParams.update(bundles.beamer_moml())



import warnings
import logging

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )


def inspect_batch(x_data, y_data, width=1.8, cmap="cividis", title=None):
    """
    Plot all given MNIST images with their corresponding labels.
    :param x_data: Numpy array of images with shape ``(b, h, w)``.
    :param y_data: Numpy array of labels with shape ``(b,)``
    :returns: Figure and axes.
    """
    num_axes = len(x_data)
    assert len(y_data) == num_axes, "Inconsistent inputs!"
    plt.rcParams.update(bundles.beamer_moml(rel_width=width))
    fig, axes = plt.subplots(ncols=num_axes)
    for i, ax in enumerate(axes):
        ax.imshow(x_data[i], cmap=cmap)
        ax.set_title(str(y_data[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def inspect_batch(x_data, y_data, width=1.8, cmap="cividis", title=None):
    """
    Plot all given MNIST images with their corresponding labels.
    :param x_data: Numpy array of images with shape ``(b, h, w)``.
    :param y_data: Numpy array of labels with shape ``(b,)``
    :returns: Figure and axes.
    """
    num_axes = len(x_data)
    assert len(y_data) == num_axes, "Inconsistent inputs!"
    plt.rcParams.update(bundles.beamer_moml(rel_width=width))
    fig, axes = plt.subplots(ncols=num_axes)
    for i, ax in enumerate(axes):
        ax.imshow(x_data[i], cmap=cmap)
        ax.set_title(str(y_data[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    return fig, axes


def inspect_batch(x_data, y_data, width=1.8, cmap="cividis", title=None):
    """
    Plot all given MNIST images with their corresponding labels.
    :param x_data: Numpy array of images with shape ``(b, h, w)``.
    :param y_data: Numpy array of labels with shape ``(b,)``
    :returns: Figure and axes.
    """
    num_axes = len(x_data)
    assert len(y_data) == num_axes, "Inconsistent inputs!"
    plt.rcParams.update(bundles.beamer_moml(rel_width=width))
    fig, axes = plt.subplots(ncols=num_axes)
    for i, ax in enumerate(axes):
        ax.imshow(x_data[i], cmap=cmap)
        ax.set_title(str(y_data[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    return fig, axes


class MNIST:
    """
    Static class to download MNIST into numpy arrays and extract a two-digit
    subset.
    """
    BASE_URL = "http://yann.lecun.com/exdb/mnist/"
    X_TRAIN_URL = "train-images-idx3-ubyte.gz"
    Y_TRAIN_URL = "train-labels-idx1-ubyte.gz"
    X_TEST_URL = "t10k-images-idx3-ubyte.gz"
    Y_TEST_URL = "t10k-labels-idx1-ubyte.gz"
    X_SHAPE = (28, 28)

    @classmethod
    def download(cls):
        """
        The MNIST dataset used in this notebook has been downloaded with this
        function. Returns a dict with the following ``np.uint8`` arrays:
        * x_train: (60000, 28, 28), y_train: (60000,)
        * x_test:  (10000, 28, 28), y_test:  (10000,)
        """
        x_train = urllib.request.urlopen(cls.BASE_URL + cls.X_TRAIN_URL).read()
        x_train = gzip.decompress(x_train)
        x_train = np.frombuffer(x_train, np.uint8, offset=16).reshape(
            -1, *cls.X_SHAPE)
        #
        y_train = urllib.request.urlopen(cls.BASE_URL + cls.Y_TRAIN_URL).read()
        y_train = gzip.decompress(y_train)
        y_train = np.frombuffer(y_train, np.uint8, offset=8)
        #
        x_test = urllib.request.urlopen(cls.BASE_URL + cls.X_TEST_URL).read()
        x_test = gzip.decompress(x_test)
        x_test = np.frombuffer(x_test, np.uint8, offset=16).reshape(
            -1, *cls.X_SHAPE)
        #
        y_test = urllib.request.urlopen(cls.BASE_URL + cls.Y_TEST_URL).read()
        y_test = gzip.decompress(y_test)
        y_test = np.frombuffer(y_test, np.uint8, offset=8)
        #
        return {"x_train": x_train, "y_train": y_train,
                "x_test": x_test, "y_test": y_test}

    @classmethod
    def extract_bmnist(cls, mnist, pos_digit=1, neg_digit=7,
                       standardize_imgs=True, dtype=np.float64):
        """
        :param mnist: The output of ``download``
        :param standardize_imgs: If true, returned images will have zero mean
          and unit variance.
        :param dtype: Ideally a large-resolution float.
        :returns: A dictionary that is a subset of the given ``mnist``, but
          only with ``pos_digit`` labeled as 1, and ``neg_digit`` labeled as 0.
        """
        # gather only desired digits, and label them +1, -1
        train_mask = (mnist["y_train"] == pos_digit) | (mnist["y_train"] ==
                                                        neg_digit)
        test_mask = (mnist["y_test"] == pos_digit) | (mnist["y_test"] ==
                                                      neg_digit)
        bmnist = {
            "x_train": mnist["x_train"][train_mask].astype(dtype),
            "y_train": ((mnist["y_train"][train_mask] == POS_DIGIT)).astype(dtype),
            "x_test": mnist["x_test"][test_mask].astype(dtype),
            "y_test": (mnist["y_test"][test_mask] == POS_DIGIT).astype(dtype)}
        # sanity check
        len_x_train, len_y_train = len(bmnist["x_train"]), len(bmnist["y_train"])
        len_x_test, len_y_test = len(bmnist["x_test"]), len(bmnist["y_test"])
        assert len_x_train == len_y_train, "Inconsistent training data in mnist?"
        assert len_x_test == len_y_test, "Inconsistent test data in mnist?"
        # optionally standardize images
        if standardize_imgs:
            bmnist["x_train"] -= bmnist["x_train"].reshape(len_x_train, -1).mean(axis=1)[:, None, None]
            bmnist["x_train"] /= bmnist["x_train"].reshape(len_x_train, -1).std(axis=1)[:, None, None]
            bmnist["x_test"] -= bmnist["x_test"].reshape(len_x_test, -1).mean(axis=1)[:, None, None]
            bmnist["x_test"] /= bmnist["x_test"].reshape(len_x_test, -1).std(axis=1)[:, None, None]
        #
        return bmnist


# Attempt to recover preexisting mnist. If not preexisting, download anew and save
#%store -r mnist
try:
    mnist
    print("Fetched MNIST from storage!")
except NameError:
    print("Downloading MNIST...")
    mnist = MNIST.download()
    #%store mnist
    
POS_DIGIT, NEG_DIGIT = 1, 7  # feel free to play around with these, but stick to (1, 7) for the submission
DTYPE = np.float64
bmnist = MNIST.extract_bmnist(mnist, POS_DIGIT, NEG_DIGIT, True, DTYPE)

inspect_samples = list(range(0, 10))
inspect_batch(bmnist["x_train"][inspect_samples], 
              bmnist["y_train"][inspect_samples])

inspect_samples = list(range(10, 20))
inspect_batch(bmnist["x_train"][inspect_samples], 
              bmnist["y_train"][inspect_samples]);

# model architecture and initialization
LAYER_SIZES = (784, 256, 64, 1)
INIT_STDDEV = 0.1
CLASSIFICATION_THRESHOLD = 0.5

# optimizer/objective
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-12

# training protocol
NUM_BATCHES = 5000
BATCH_SIZE = 25
RANDOM_SEED = 12345

def train_dataloader(bmnist, batch_size=50, rng=jax.random.PRNGKey(12345)):
    """
    Given a binary MNIST dataset, this generator runs infinitely, returning
    randomized batches from the training split.
    
    :param bmnist: Dictionary as returned by ``MNIST.extract_bmnist``
    :param rng: If a ``jax```random key is given, use it to shuffle
      all entries.
    :yields: An input-output pair of numpy arrays ``(x, y)``, where
      the first dimension of the arrays equals ``batch_size``,
      except for the last batch that may be smaller.
    """
    len_train = len(bmnist["x_train"])
    while True:
        rng = jax.random.split(rng)[1]
        perm = jax.random.permutation(rng, len_train)
        for i in range(0, len_train, batch_size):
            x = bmnist["x_train"][perm[i : (i + batch_size)], ...]
            y = bmnist["y_train"][perm[i : (i + batch_size)], ...]
            yield (x, y)


def test_dataloader(bmnist, batch_size=50):
    """
    Given a binary MNIST dataset, this generator runs once over its
    test split, in batched manner.
    
    :param bmnist: Dictionary as returned by ``MNIST.extract_bmnist``
    :yields: An input-output pair of numpy arrays ``(x, y)``, where
      the first dimension of the arrays equals ``batch_size``,
      except for the last batch that may be smaller.
    """
    assert batch_size > 0, "batch_size <= 0 not supported"
    for i in range(0, len(bmnist["x_test"]), batch_size):
        x = bmnist["x_test"][i : (i + batch_size), ...]
        y = bmnist["y_test"][i : (i + batch_size), ...]
        yield (x, y)


def loss_fn(params, inputs, targets, l2_reg=0.0):
    """
    :param params: Network parameters. See ``mlp`` docstring.
    :param inputs: Batch of network inputs. See ``mlp`` docstring.
    :param targets: Batch of ground truth annotations corresponding to ``inputs``,
      as provided by the dataloader.
    :param l2_reg: Strength of the L2 regularization term, such that
      ``result = cross_entropy + (0.5 * l2_reg * l2norm(params)**2)``.
    :returns: A single scalar representing the empirical risk plus the L2 
      regularizer over the given batch, with respect to the given parameters.
    """
    # ERM via cross-entropy
    preds = mlp(params, inputs) # this returns log probabilities
    result = jnp.logaddexp(0, -preds * (2 * targets - 1)).mean()    
    # L2 regularization
    reg = 0.5 * l2_reg * sum(jnp.sum(w ** 2) + jnp.sum(b ** 2) for w, b in params)
    result = result + reg
    #
    return result


#%%
def mlp(params, inputs, nonlinearity=jax.nn.relu):
    """
    Computes the forward pass of an MLP, defined using JAX components. Note that
    it returns the *logits*. To map logits into predicted scores, a sigmoid
    function can be applied.
    
    :param params: List of pairs in the form ``[(w1, b1), (w2, b2), ...]`` where
      ``w_i, b_i`` are the weights and biases for layer ``i``, such that a layer
      computes ``outputs = nonlinearity((w_i @ inputs) + b_i)``.
    :param inputs: Batch of flattened input images with shape ``(batch, in_shape)``
    :returns: A vector of shape ``(batch,)``, containing one logit per input that
      should predict the corresponding binary class.
    """
    result = inputs
    for w_i, b_i in params[:-1]:
        #print("w_i", w_i.shape)
        #print("b_i", b_i.shape)
        result = nonlinearity(jnp.dot(result, w_i) + b_i)
    #
    #print("final_w", jnp.array(params[-1][0]).shape)
    #print("final_b", jnp.array(params[-1][1]).shape)
    final_w, final_b = params[-1]
    logits = jnp.dot(result, final_w) + final_b
    return logits[..., 0]

#%%
def create_mlp_params(layer_sizes, stddev=0.1, rng=jax.random.PRNGKey(12345)):
    """
    Creates MLP parameters of given sizes and initializes them with Gaussian
    noise of zero mean and given standard deviation.
    :param layer_sizes: List of integers in the form ``[d1, d2, ...]``,
      where each MLP layer maps from ``d_i`` dimensions to ``d_{i+1}``.
    :param stddev: Standard deviation of the initial Gaussian noise.
    :param rng: ``jax.random.PRNGKey`` to draw noise from.
    """
    params = []
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, rng_b = jax.random.split(rng)
        w = jax.random.normal(rng, (m, n)) * stddev
        b = jax.random.normal(rng_b, (n,)) * stddev
        params.append((w, b))
    return params


def test_predictions(params, bmnist, batch_size=50, threshold=0.5):
    """
    Helper function to run ``sigmoid(model)`` over the whole test subset
    and compute the accuracy.
    
    :param threshold: Any sigmoid outputs above this number will be consider
      positive (i.e. a value of 1), otherwise negative (i.e. a value of 0).
    :param bmnist: See ``test_dataloader``.
    :param batch_size: See ``test_dataloader``.
    :returns: The triple ``(accuracy, logits, targets)``, where 
      ``accuracy`` is the ratio of correctly classified samples, ``logits``
      are the predicted logits following the order provided by
      ``test_dataloader``, and ``targets`` are the corresponding ground
      truth annotations.
    """
    all_logits = []
    targets = []
    for x_batch, y_batch in test_dataloader(bmnist, batch_size):
        logits = mlp(params, x_batch.reshape(len(x_batch), -1))
        all_logits.extend(list(logits))
        targets.extend(list(y_batch))
    #
    predictions = jax.nn.sigmoid(np.array(all_logits)) > threshold
    targets = jnp.array(targets)
    accuracy = (predictions == targets).sum() / len(predictions)
    return accuracy, all_logits, targets

# 1. Dataloaders
train_dl = train_dataloader(bmnist, BATCH_SIZE, rng=jax.random.PRNGKey(RANDOM_SEED))
test_dl = test_dataloader(bmnist, BATCH_SIZE)

# 2. Model params
mlp_params = create_mlp_params(LAYER_SIZES, INIT_STDDEV, 
                               jax.random.PRNGKey(RANDOM_SEED))

# 3. Optimizer and JIT update step
opt_init, opt_update, get_params = jopt.sgd(LEARNING_RATE)
opt_state = opt_init(mlp_params)

@jax.jit
def update(step, opt_state, inputs, targets, l2_reg=0.0):
    """
    In order to speed up computations (not really necessary for small
    examples like this one, but crucial for larger DL setups), we 
    "bundle" the forwardprop, backprop and update steps into a single
    JIT-able function.
    """
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state), 
                                               inputs, targets, l2_reg)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


# Training loop
losses, test_accs = [], []  # we will gather losses and accuracies
t0 = time()
#
for batch_t, (x_batch, y_batch) in enumerate(train_dl, 1):
    if batch_t > NUM_BATCHES:
        break
    #
    x_batch = x_batch.reshape(len(x_batch), -1)
    loss, opt_state = update(batch_t, opt_state, x_batch, y_batch, WEIGHT_DECAY)
    losses.append(loss)
    if batch_t % 200 == 0:
        test_acc, _, _ = test_predictions(get_params(opt_state), bmnist,
                                       BATCH_SIZE, CLASSIFICATION_THRESHOLD)
        print(f"[step {batch_t:07d}] Loss={loss:5f}, Test accuracy={test_acc:2f}")
        test_accs.append((batch_t, test_acc))
#
print("Elapsed seconds:", time() - t0)

# Once trained, gather the MAP params to use later!
MAP_PARAMS = get_params(opt_state)

plt.rcParams.update(bundles.beamer_moml(rel_width=1.8, rel_height=1.5))
fig, (ax_loss, ax_acc) = plt.subplots(nrows=2)
#
ax_loss.plot(range(NUM_BATCHES), losses)
ax_loss.set_title("Loss")
#
ax_acc.plot(*zip(*test_accs))
_ = ax_acc.set_title("Test Accuracy")

class LossFnWrapper:
    """
    Curried wrapper of ``loss_fn`` that first fixes inputs, targets
    and L2 regularization scale, and then accepts model parameters
    as an extended list of parameters, instead of a single parameter.
    This allows us to tell ``jax`` that we want to differentiate
    with respect to a subset of the parameters instead of all of them.
    
    Usage example::
    
      lfn = LossFnWrapper(xtest_data, ytest_data, 1e-10)
      loss = lfn(mlp_params)  # This is the __call__ method
      
    .. note::
      Last week we mentioned that ``jax`` is functional and we should
      not be passing stateful objects like class instances. But note
      that this class is nothing else but the realization of a
      perfectly valid 'currying' functional software pattern when
      used like in the example above. This kind of class is typically
      called a 'callable' or a 'functor'.
    """
    
    def __init__(self, inputs, targets, l2_reg=0.0):
        """
        :param inputs: See ``loss_fn`` docstring.
        :param targets: See ``loss_fn`` docstring.
        :param l2_reg: See ``loss_fn`` docstring.
        """
        self.inputs = inputs
        self.targets = targets
        self.l2_reg = l2_reg
    
    def __call__(self, *params):
        """        
        :param params: The MLP parameters as given to ``loss_fn``, but
          provided as separate entries instead of a single collection.
        :returns: The loss, same as ``loss_fn``.
        """
        return loss_fn(params, self.inputs, self.targets, self.l2_reg)

#%% Recompute Hessian
def compute_hessian_lastlayer(params, bmnist):
    """
    This function computes the Hessian of ``loss_fn`` with respect to
    the last-layer parameters, over the test subset of ``bmnist``.
    
    Note that in a rigorous/more complex scenario we would compute a
    running average or a subsample of the training subset, and not
    the test subset.
    
    :param params: List of pairs in the form ``[(w1, b1), (w2, b2), ...]``
      where ``w_i, b_i`` are the weights and biases for layer ``i``, such
      that a layer computes ``outputs = nonlinearity((w_i @ inputs) + b_i)``.
    :param bmnist: Dictionary as returned by ``MNIST.extract_bmnist``
    :returns: A square, symmetric matrix corresponding to the last-layer
      Hessian, i.e. the last ``(w_i, b_i)`` entry from ``params``.
    """
    test_inputs = bmnist["x_test"].reshape(len(bmnist["x_test"]), -1)
    test_targets = bmnist["y_test"]
    repackaged_params = params.copy() 

    def loss_wrapper(last_layer_params):
        w = last_layer_params[:-1].reshape(last_layer_params[:-1].shape[0], 1)
        b = jnp.array([last_layer_params[-1]])
        _repackaged_params = repackaged_params.copy()
        _repackaged_params[-1] =  (w,b)
        for idx, mp in enumerate(repackaged_params):
            assert repackaged_params[idx][0].shape == _repackaged_params[idx][0].shape , f"idx {idx} {repackaged_params[idx][0].shape} != {_repackaged_params[idx][0].shape}"
            assert repackaged_params[idx][1].shape == _repackaged_params[idx][1].shape 
        return loss_fn(_repackaged_params, test_inputs, test_targets, 1e-10) 

    last_params = jnp.append(*repackaged_params[-1])
    H_matrix = jax.hessian(loss_wrapper)(last_params)
    
    return H_matrix 

H_matrix = compute_hessian_lastlayer(MAP_PARAMS, bmnist)




#%%
# compute last-layer Hessian
H_matrix = compute_hessian_lastlayer(MAP_PARAMS, bmnist)

# compute the Hessian eigendecomposition (ew=eigenwert, ev=eigenvector)
H_ews, H_evs = jnp.linalg.eigh(H_matrix)
Hi = H_evs @ jnp.diag(1 / H_evs) @ H_evs.T

# plot the Hessian and spectrum
plt.rcParams.update(bundles.beamer_moml(rel_height=1.5))
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
im1 = ax1.imshow(H_matrix)
ax1.set_title(f"Last-layer Hessian.\nShape: {H_matrix.shape}\nDatatype:{H_matrix.dtype}")
#
ax2.plot(H_ews)
ax2.set_title(f"Last-layer Hessian eigenvalues")
#
fig.colorbar(im1, ax=ax1);
#%%
def map_confidence(params, bmnist, batch_size=50):
    """
    This function computes and returns ``sigmoid(abs(model))`` over the
    whole test subset, i.e. the MAP confidence scores typically used in
    DL that exhibit pathological overconfidence.
    
    :param params: Parameters for the ``mlp`` model.
    :param bmnist: Dictionary as returned by ``MNIST.extract_bmnist``.
    :param batch_size: To prevent running out of memory or going too slow,
      result is computed in batches of this size.
    :returns: One confidence scalar per entry in the test subset,
      corresponding to ``sigmoid(abs(mlp(x)))``.
    """
    result = []
    for x_batch, y_batch in test_dataloader(bmnist, batch_size):
        x_batch = x_batch.reshape(len(x_batch), -1)
        logits = mlp(params, x_batch)
        confidences = jax.nn.sigmoid(abs(logits))
        result.extend(confidences)
    #
    result = jnp.array(result)
    return result


@jax.jit
def compute_jacobian_lastlayer(params, inputs):
    """
    This function computes the Jacobian of ``mlp`` with respect to
    the last-layer parameters, given the ``inputs``.
    
    :param params: See ``compute_hessian_lastlayer``.
    :param inputs: Batch of flattened input images with shape 
      ``(batch, in_shape)``.
    :returns: A Jacobian matrix of shape ``(batch, lastlayer_params)``,
      containing the partial derivatives of the ``mlp`` outputs with
      respect to the last-layer parameters.
    """
    test_inputs  = inputs #bmnist["x_test"].reshape(len(bmnist["x_test"]), -1)

    def mlp_output_wrapper(last_layer_params, inputs):
        w = last_layer_params[:-1].reshape(last_layer_params[:-1].shape[0], 1)
        b = jnp.array([last_layer_params[-1]])
        _repackaged_params = params.copy()
        _repackaged_params[-1] =  (w,b)

        for idx, mp in enumerate(params):
            assert params[idx][0].shape == _repackaged_params[idx][0].shape , f"idx {idx} {repackaged_params[idx][0].shape} != {_repackaged_params[idx][0].shape}"
            assert params[idx][1].shape == _repackaged_params[idx][1].shape 
        mlp_out = mlp(_repackaged_params, inputs) 
        return mlp_out 

    last_params = jnp.append(*params[-1])
    J_vec = jax.jacobian(mlp_output_wrapper)(last_params, inputs)
    return J_vec


def la_predictive_confidence(params, bmnist, h_ews, h_evs, 
                             prior_precision=1, batch_size=50):
    """
    This function computes and returns ``sigmoid(abs(z(x)))`` over the
    whole test subset, i.e. the LA predictive confidence as derived
    above (note that our computed Hessian is already ``-psi``).
    
    :param params: see ``map_confidence``.
    :param bmnist: see ``map_confidence``.
    :param batch_size: see ``map_confidence``.
    :param h_ews: Vector of eigenvalues obtained via Hessian eigendecomposition.
    :param h_evs: Orthogonal square matrix of eigenvectors obtained via
      Hessian eigendecomposition, such that ``H = h_evs @ h_ews @ h_evs.T``
    :param prior_precision: Since we are inverting ``H``, and ``h_ews`` tend to
      be near-zero, we may run into numerical issues. This scalar is a constant
      being added to all eigenvalues, preventing this issue and also becoming
      a regularization hyperparameter.
    :returns: One confidence scalar per entry in the test subset, correspoding
      to the LA predictive confidence ``sigmoid(abs(z(x)))``.
    """
    neg_psi  = compute_hessian_lastlayer(params, bmnist, batch_size=batch_size)
    
    result = []

    for x_batch, y_batch in test_dataloader(bmnist, batch_size):
        x_batch = x_batch.reshape(len(x_batch), -1)
        logits = mlp(params, x_batch) # f(x', \theta*)
        j_vec = compute_jacobian_lastlayer(params, x_batch) 
        1 + (jnp.pi / 8.0) * j_vec.T @ jnp.linalg.inv(neg_psi) @ j_vec
        # x batchwill be 50 
        #confidences = jax.nn.sigmoid(abs(logits))

        result.extend(confidences)
    #
    result = jnp.array(result)
    # raise NotImplementedError("TODO")

#%% Compute Last Layer Jacobian.
J = compute_jacobian_lastlayer(MAP_PARAMS, bmnist['x_test'].reshape(len(bmnist['x_test']), -1))
#%%
# compute MAP and LA predictive confidences over the test set for different precision hyperpars
PRIOR_PRECISIONS = [1, 3, 10, 30, 100]
MAP_CONFS = map_confidence(MAP_PARAMS, bmnist)
#%%
LAP_CONFS = {pp: la_predictive_confidence(MAP_PARAMS, bmnist, H_ews, H_evs, prior_precision=pp)
             for pp in PRIOR_PRECISIONS}

# plot 
plt.rcParams.update(bundles.beamer_moml(rel_height=2))
fig, ax = plt.subplots(figsize=(10, 5))
#
ax.plot(sorted(MAP_CONFS), label="MAP")
for pp, confs in LAP_CONFS.items():
    ax.plot(sorted(confs), label=f"LA predictive (precision: {pp})")
ax.legend()
fig.suptitle("binary MNIST test set confidences\nsorted in ascending order");
#%%

