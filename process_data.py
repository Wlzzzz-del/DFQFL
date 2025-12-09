import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
# from utils import angle_encode_data

def generate_dataset(ds_name,digits_to_keep, n_samples=2000):
    """
    Generates and preprocesses the dataset based on the specified parameters.
    modify by https://github.com/positivetechnologylab/Quorus
    Parameters:
      ds_name: Name of the dataset to load ('mnist' or 'fashion-mnist').
      digits_to_keep: List of digit classes to retain in the dataset.
      n_samples: Optional integer specifying the number of samples to subsample.
    Returns:
      A tuple (tensor_x, tensor_y) where tensor_x is the input data as a PyTorch Tensor
    """

    if ds_name in {"mnist", "mnist-784"}:
        is_fashion = False
        dataset_cls = datasets.MNIST
    elif ds_name in {"fashion-mnist", "fashion mnist", "fashion"}:
        is_fashion = True
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion-mnist'.")

    transform = transforms.Compose([transforms.ToTensor()])

    train_data_raw = dataset_cls(root='./data', train=True, download=True, transform=transform)
    test_data_raw = dataset_cls(root='./data', train=False, download=True, transform=transform)

    X_train = train_data_raw.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_data_raw.targets.numpy()

    X_test = test_data_raw.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_data_raw.targets.numpy()

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).astype(int)

    keep_ids = np.array(digits_to_keep) 
    mask = np.isin(y, keep_ids)
    X, y = X[mask], y[mask]

    mapping = {label: idx for idx, label in enumerate(keep_ids)}
    y = np.array([mapping[int(lbl)] for lbl in y], dtype=int)

    if n_samples is not None and n_samples < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=42
        )

    tensor_x = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y).long()

    print(f"LOG: Data loaded: {ds_name}, Shape: {X.shape}, Classes: {len(keep_ids)}")

    return tensor_x,tensor_y

"""## Split data federated function"""
def split_data_federated(X, y, client_config, test_frac, val_frac=0.2,
                         feature_skew=1.0, label_skew=None, random_state=42, local_pca=False, do_lda=False, feat_sel_type="top", amp_embed=False, feat_ordering="same",
                         shared_pca=False, fed_pca_mocked=False):
  """
    Splits data for federated learning, inducing controllable label‐skew
    that goes from uniform (label_skew=1) to linear (s=0.5) to extreme (s→0).
    modify by https://github.com/positivetechnologylab/Quorus

    Parameters:
      X: The input data to split among clients.
      y: The input labels to split among clients.
      client_config: The client configuration dictionary used to determine the percentage of data each client.
      test_frac: Float representing the fraction of the input data used for testing.
      val_frac: Float representing the fraction of each clients' data used for validation.
      feature_skew: Float representing the strength of the skew that each client faces in terms of feature values.
      label_skew: None or float in [0,1].
        - 1.0 => each client’s labels are uniform.
        - 0.5 => exactly linear descending (green line).
        - 0.0 => extreme “red” skew (almost all mass on class 0).
        - Values in between smoothly interpolate via exponent θ = (1−s)/s.
      random_state: Integer representing the random state used for the entire program 
      local_pca: Boolean indicating whether or not PCA should be performed locally.
      do_lda: Boolean indicating whether or not random sketching should be performed.
      feat_sel_type: String representing the choice of features that the client will take later.
      amp_embed: Boolean indicating whether or not the data will be amplitude encoded.
      feat_ordering: String representing the ordering of features to sample in the case where data is amplitude encoded 

    Returns:
      clients_data: a dictionary mapping integers representing client types to a list of data for each client, where the i-th element of the list is the data for
      client i, and each client has a list of data in the form [(X_train, y_train), (X_val, y_val), (pca_obj, pca_reduced_data)]
      with the third element only being present if local PCA is performed
      (X_test, y_test): a tuple of testing data and labels

  """
  X = np.array(X)
  y = np.array(y)
  if random_state is not None:
      np.random.seed(random_state)

  # 1) global train/test split
  n = len(X)
  perm = np.random.permutation(n)
  tsize = int(test_frac * n)
  test_idx, train_idx = perm[:tsize], perm[tsize:]
  X_test,  y_test  = X[test_idx],  y[test_idx]
  X_train, y_train = X[train_idx], y[train_idx]
  n_train = len(X_train)

  rel_idx = np.arange(n_train)
  pointer = 0
  clients_data = {}

  max_cli_size = max(client_config.keys())

  if do_lda:
    sketch_mat = np.random.normal(loc=0.0, scale=1/np.sqrt(X.shape[1]), size=(X.shape[1], max_cli_size))

  # sanity
  total_pct = sum(cfg["percentage_data"] for cfg in client_config.values())
  if total_pct > 1.0:
      raise ValueError("Sum of percentage_data > 1")

  # 2) per-client-type allocation
  for cnum, cfg in client_config.items():
      pct, n_clients = cfg["percentage_data"], 1

      # carve off this type’s pool
      alloc_n = int(pct * n_train)
      alloc_n = min(alloc_n, n_train - pointer)
      alloc_idx = rel_idx[pointer : pointer + alloc_n]
      pointer += alloc_n

      # --- feature skew ---
      feats = X_train[alloc_idx, 0].astype(float)
      if feats.size:
          lo, hi = feats.min(), feats.max()
          norm_feat = (feats - lo)/(hi - lo) if hi > lo else np.zeros_like(feats)
      else:
          norm_feat = feats
      rand_comp = np.random.rand(len(alloc_idx))
      scores = feature_skew*norm_feat + (1-feature_skew)*rand_comp
      alloc_idx = alloc_idx[np.argsort(scores)]

      # --- label skew --- (保持不变)
      if label_skew is None:
          client_chunks = np.array_split(alloc_idx, n_clients)
      else:
          labels_sorted = np.array(sorted(np.unique(y_train[alloc_idx])))
          C = len(labels_sorted)
          s = float(label_skew)
          theta = (1.0 - s)/s if s>0 else np.inf
          ranks = np.arange(C, 0, -1, dtype=float)
          w = (ranks**theta) if np.isfinite(theta) else np.zeros_like(ranks)
          if not np.isfinite(theta): w[0] = 1.0
          p_local = w / w.sum()
          remaining = np.array(alloc_idx, dtype=int)
          lbl2idx = {lbl:i for i,lbl in enumerate(labels_sorted)}
          client_chunks = []
          for client_id in range(n_clients):
              n_rem = len(remaining)
              if n_rem == 0:
                  client_chunks.append(np.array([],dtype=int)); continue
              clients_left = n_clients - client_id
              take_n = int(math.ceil(n_rem / clients_left))
              classes_rem = y_train[remaining]
              q = np.array([p_local[lbl2idx[l]] for l in classes_rem])
              q /= q.sum()
              sel_idx = np.random.choice(n_rem, size=min(take_n, n_rem), replace=False, p=q)
              sel = remaining[sel_idx]
              client_chunks.append(sel)
              remaining = np.delete(remaining, sel_idx)

      # 3) split each client chunk into train/val
      clients_data[cnum] = []
      for chunk in client_chunks:
          m = len(chunk)
          v = int(val_frac * m)
          client_data_chunk = X_train[chunk]
          val_idx   = chunk[:v]
          train_idx = chunk[v:]
          n_tot_comps = cnum
          if feat_sel_type != "top" or shared_pca:
            n_tot_comps = max_cli_size

          # Perform local PCA (and local random sketching) if specified.

          client_data_chunk, pca, client_data_chunk_pca = client_data_chunk, None, None

          # If the data will be amplitude encoded...
          if amp_embed and feat_ordering == "highest_var":
            variances = X.var(axis=0, ddof=0)
            order = np.argsort(variances)[::-1]
            client_data_chunk = client_data_chunk[:, order] + 1e-3

          # Store the original data for this particular client.
          if shared_pca:
            client_data_chunk = X_train[chunk]

          X_train_data = client_data_chunk[v:]
          X_val_data = client_data_chunk[:v]

          X_train_tensor = torch.tensor(X_train_data).float()
          y_train_tensor = torch.tensor(y_train[train_idx]).long()

          X_val_tensor = torch.tensor(X_val_data).float()
          y_val_tensor = torch.tensor(y_train[val_idx]).long()

          client_data_lst = [
              (X_train_tensor, y_train_tensor), # 训练集：(Tensor, Tensor)
              (X_val_tensor, y_val_tensor)      # 验证集：(Tensor, Tensor)
          ]

          if local_pca:
            client_data_lst.append((pca, client_data_chunk_pca))

          clients_data[cnum].append(
              client_data_lst
          )

  # 转换全局测试集为 Tensor，以便在 PyTorch 中使用
  X_test_tensor = torch.tensor(X_test).float()
  y_test_tensor = torch.tensor(y_test).long()

  return clients_data, (X_test_tensor, y_test_tensor)


def generate_client_percentage_config(num_clients):
    """
    generated normalized client percentage config for federated learning.
    modify by https://github.com/positivetechnologylab/Quorus

    :param num_clients: number of clients (int)
    :return: config
    """
    share = 1.0 / num_clients

    clients_config_arg = {
        i: {"percentage_data": share} 
        for i in range(num_clients)
    }

    return clients_config_arg