import torch

def masked_mean(x, mask, dim=None, keepdim=False, return_percentage=False):
    x = x.masked_fill(mask==0, 0)
    mask_sum = mask.sum(dim=dim, keepdim=keepdim)
    mask_sum[mask_sum==0] = 1.
    x_mean = x.sum(dim=dim, keepdim=keepdim) / mask_sum

    if return_percentage:
        return x_mean, mask.type(x_mean.dtype).mean(dim=dim)
    return x_mean

def masked_std(x, mask, dim=None, keepdim=False, return_percentage=False, eps=1e-4):
    x = x.masked_fill(mask==0, 0)
    mask_sum = mask.sum(dim=dim, keepdim=True)
    mask_sum[mask_sum==0] = 1.
    x_mean = x.sum(dim=dim, keepdim=True) / mask_sum

    x = ((x - x_mean)**2).masked_fill(mask==0, 0)
    x_std = torch.sqrt(x.sum(dim=dim, keepdim=True) / mask_sum + eps)
    if not keepdim:
        x_std = x_std.squeeze(dim)

    if return_percentage:
        return x_std, mask.type(x_mean.dtype).mean(dim=dim)
    return x_std

def torch_nanmean(x, dim=None, keepdim=False, return_percentage=False):
    return masked_mean(x, ~torch.isnan(x), dim=dim, keepdim=keepdim, return_percentage=return_percentage)

def torch_nanstd(x, dim=None, keepdim=False, return_percentage=False, eps=1e-5):
    return masked_std(x, ~torch.isnan(x), dim=dim, keepdim=keepdim, return_percentage=return_percentage, eps=eps)

def clip_outliers(data, dim=1, mask=None, n_sigma=4):
    if mask is None:
        mask = ~torch.isnan(data)
    data_mean = masked_mean(data, mask, dim=dim, keepdim=True)
    data_std = masked_std(data, mask, dim=dim, keepdim=True)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data = torch.max(-torch.log(1+torch.abs(data)) + lower, data)
    data = torch.min(torch.log(1+torch.abs(data)) + upper, data)
    return data

def normalize_data(data, dim=1, mask=None):
    if mask is None:
        mask = ~torch.isnan(data)
    mean = masked_mean(data, mask, dim=dim, keepdim=True)
    std = masked_std(data, mask, dim=dim, keepdim=True)

    data = (data - mean) / std
    return data

def process(xs, ys, dim=1, mask=None, classification=False):
    # clip outliers
    xs = clip_outliers(xs, dim=dim, mask=mask)
    xs = normalize_data(xs, dim=dim, mask=mask)

    if classification:
        ys = ys.to(torch.long)
    else:
        ys = clip_outliers(ys, dim=dim)
        ys = normalize_data(ys, dim=dim)

    return xs, ys

def process_data(data, classification=False):
    x_train, y_train, x_test, y_test = data
    """
    x_train: (train_size, sequence_length)
    y_train: (train_size,)
    x_test: (test_size, sequence_length)
    y_test: (test_size,)
    """
    n_test = x_test.shape[0]

    xs = torch.cat((x_train, x_test), dim=0)
    ys = torch.cat((y_train, y_test), dim=0)
    xs, ys = process(xs, ys, dim=0, classification=classification)

    return (
        xs[:-n_test, :],
        ys[:-n_test],
        xs[-n_test:, :],
        ys[-n_test:]
    )
