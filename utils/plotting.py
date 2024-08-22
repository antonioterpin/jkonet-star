import os
from pathlib import Path
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import yaml

def plot_couplings(data):
    """
    Plots circles at x coordinates, crosses at y coordinates,
    and connects them with lines whose widths are proportional to weights w.

    Args:
    data (np.ndarray): An array of shape (n, 5) where each row contains:
                       x0, x1 (coordinates of the circle), 
                       y0, y1 (coordinates of the cross), 
                       w (weight for line width).
    """
    # Extract coordinates and weights
    weights = data[:, -1]
    x_coords = data[:, :(data.shape[1] - 1) // 2]
    y_coords = data[:, (data.shape[1] - 1) // 2:-1]

    # Normalize weights for line width between a minimum and a maximum
    line_widths = 2 * (weights / weights.max())  # Normalize and scale line width by weight

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot circles at x positions
    ax.scatter(x_coords[:, 0], x_coords[:, 1], c='blue', s=100, edgecolors='black', label='Circles', marker='o')

    # Plot crosses at y positions
    ax.scatter(y_coords[:, 0], y_coords[:, 1], c='red', s=100, label='Crosses', marker='x')

    # Draw lines connecting circles and crosses
    for x, y, lw in zip(x_coords, y_coords, line_widths):
        ax.plot([x[0], y[0]], [x[1], y[1]], 'gray', linewidth=lw)

    # Adding labels and title for clarity
    # ax.set_xlabel('X coordinate')
    # ax.set_ylabel('Y coordinate')
    # ax.set_title('Connections Between Points')
    # ax.legend()

    return fig, ax


def domain_from_data(data):
    # set max and min values
    x_min = np.amin(data, axis=0)[:, 0].min() - 2.0
    x_max = np.amax(data, axis=0)[:, 0].max() + 2.0

    y_min = np.amin(data, axis=0)[:, 1].min() - 2.0
    y_max = np.amax(data, axis=0)[:, 1].max() + 2.0

    return ((x_min, y_min), (x_max, y_max))


def grid_from_domain(domain, n_samples=100):
    # create grid
    x, y = np.meshgrid(np.linspace(domain[0][0], domain[1][0], n_samples),
                       np.linspace(domain[0][1], domain[1][1], n_samples))
    grid = np.vstack((x.ravel(), y.ravel())).T

    if len(domain[0]) > 2:
        grid = np.concatenate((grid, np.zeros((grid.shape[0],
                                               len(domain[0]) - 2))), axis=1)

    return x, y, grid


def plot_level_curves(function, domain, n_samples=100, dimensions=2, save_to=None):
    f = jax.vmap(function)
    x, y, grid = grid_from_domain(domain, n_samples)

    # get values
    if grid.shape[1] < dimensions:
        v = np.concatenate((grid, np.zeros((grid.shape[0],
                                            dimensions - grid.shape[1]))), axis=1)
        pred = f(v)
    else:
        pred = f(grid)
    z = pred.reshape(x.shape)

    # plot energy predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(domain[0][0], domain[1][0]), ylim=(domain[0][1], domain[1][1]))
    ax.grid(False)

    ax.contour(x, y, z, levels=15, linewidths=.5, linestyles='dotted',
               colors='k')
    ctr = ax.contourf(x, y, z, levels=15, cmap='Blues')
    
    if save_to is not None:
        # Save the data to a text file
        path = Path(save_to)
        os.makedirs(path.parent.absolute(), exist_ok=True)
        file = open(save_to, 'w')
        file.close()
        np.savetxt(save_to, np.column_stack(
            (x.flatten(), y.flatten(), z.flatten())), fmt='%-7.2f')


    fig.colorbar(ctr, ax=ax)

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to + '.png')
    return fig


def plot_predictions(predicted, data_dict, interval, model, save_to=None, n_particles=200):
    if interval is None:
        start, end = 0, max(data_dict.keys())
    else:
        start, end = interval

    filtered_timesteps = range(start, end + 1)
    data = np.zeros((len(filtered_timesteps), n_particles, predicted.shape[2]))

    # set max and min values
    data = data[:, :n_particles, :]
    predicted = predicted[:, :n_particles, :]
    for i, t in enumerate(filtered_timesteps):
        if t in data_dict:
            data[i, :, :] = data_dict[t][:n_particles, :]

    x_min = np.min((np.amin(data, axis=0)[:, 0].min(), np.amin(predicted, axis=0)[:, 0].min())) - 2.0
    x_max = np.max((np.amax(data, axis=0)[:, 0].max(), np.amax(predicted, axis=0)[:, 0].max())) + 2.0

    y_min = np.min((np.amin(data, axis=0)[:, 1].min(), np.amin(predicted, axis=0)[:, 1].min())) - 2.0
    y_max = np.max((np.amax(data, axis=0)[:, 1].max(), np.amax(predicted, axis=0)[:, 1].max())) + 2.0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    colors = yaml.safe_load(open('style.yaml'))
    c_data = clr.LinearSegmentedColormap.from_list(
        'Greys', [colors['groundtruth']['light'], colors['groundtruth']['dark']],
        N=data.shape[0])
    c_pred = clr.LinearSegmentedColormap.from_list(
        'Blues', [colors[model]['light'], colors[model]['dark']], N=predicted.shape[0])

    for t in range(data.shape[0]):
        x, y = data[t][:, 0], data[t][:, 1]
        ax.scatter(x, y, edgecolors=[c_data(t)],
                   facecolor='none', label='data, t={}'.format(t), marker=colors['groundtruth']['marker'])
        if save_to is not None:
            np.savetxt(save_to + f'-data-{t}.txt', np.column_stack(
                (x.flatten(), y.flatten())), fmt='%-7.2f')

    for t in range(predicted.shape[0]):
        x, y = predicted[t][:, 0], predicted[t][:, 1]
        ax.scatter(x, y, c=[c_pred(t)],
                   label='predicted, t={}'.format(t), marker=colors[model]['marker'])
        if save_to is not None:
            np.savetxt(save_to + f'-predicted-{t}.txt', np.column_stack(
                (x.flatten(), y.flatten())), fmt='%-7.2f')

    ax.legend(bbox_to_anchor=(0.5, 1.25), fontsize='medium',
              loc='upper center', ncol=3,
              columnspacing=1, frameon=False)

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to + '.png')
    return fig

def colormap_from_config(config):
    light = config['light']
    dark = config['dark']
    return clr.LinearSegmentedColormap.from_list('custom', [light, dark])

def plot_heatmap(X, Y, Z, labels, title, colormap, save_to=None):
    fig = plt.figure()
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=colormap)
    plt.colorbar(label=labels['Z'])
    plt.xlabel(labels['X'])
    plt.ylabel(labels['Y'])
    plt.xticks(X[0])
    plt.yticks(Y[:, 0])
    plt.title(title)

    if save_to is not None:
        plt.savefig(save_to + '.png')
        with open(save_to + '.csv', 'w') as file:
            file.write(f'x y z\n')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    file.write(f'{X[i, j]:.10f} {Y[i, j]:.10f} {Z[i, j]:.10f}\n')
                file.write('\n')
    return fig

def plot_boxplot_comparison_models(data, model_names, title, save_to=None, yscale='linear'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bp = ax.boxplot(data, patch_artist=True, meanline=True, showmeans=True)

    # Customizing the box plot colors
    style = yaml.safe_load(open('style.yaml'))
    colors = [style[model]['dark'] for model in model_names]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Adding custom labels for clarity
    ax.set_xticklabels(model_names)
    ax.set_title(title)
    ax.set_ylabel('Execution Time [s]')
    ax.set_yscale(yscale)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to + '.png')
        for model in model_names:
            np.savetxt(save_to + f'-{model}.txt', data[model_names.index(model)], fmt='%.2f')

    return fig
    

def plot_comparison_models(error1, error2, labels, model_names, title, save_to=None, cmaps=None, insert_inset=False, size=100):
    error1 = np.asarray(error1)
    error2 = np.asarray(error2)
    labels = np.asarray(labels)

    #Normalize errors
    max_error = np.nanmax(np.concatenate((error1, error2)))
    normalized_error1 = error1 / max_error
    normalized_error2 = error2 / max_error

    #Group them by labels
    unique_labels = np.unique(labels)
    if cmaps is None:
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        cmaps = [cmap(i) for i in range(len(unique_labels))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    any_nan_x = False
    any_nan_y = False
    if insert_inset:
        ax_inset = fig.add_axes([0.4, 0.35, 0.45, 0.45])
        max_x, max_y = 0, 0
        min_x, min_y = 0, 0
    for i, label in enumerate(unique_labels):
        label_err1 = normalized_error1[labels == label]
        label_err2 = normalized_error2[labels == label]
        mean_err1 = np.mean(label_err1)
        mean_err2 = np.mean(label_err2)
        
        any_nan_x = any_nan_x or np.isnan(label_err1).any()
        any_nan_y = any_nan_y or np.isnan(label_err2).any()

        # Replace NaN values with 1.2
        label_err1 = np.nan_to_num(label_err1, nan=1.2)
        label_err2 = np.nan_to_num(label_err2, nan=1.2)

        if insert_inset:
            max_x = max(max_x, np.max(label_err1))
            min_x = min(min_x, np.min(label_err1))
            max_y = max(max_y, np.max(label_err2))
            min_y = min(min_y, np.min(label_err2))

        ax.scatter(
            label_err1, label_err2, label=label, 
            alpha=0.5, color=cmaps[i], s=size)
        ax.scatter(
            mean_err1, mean_err2, label=label, 
            alpha=1, color=cmaps[i], s=size)
        
        if insert_inset:
            ax_inset.scatter(label_err1, label_err2, label=label, alpha=0.5, color=cmaps[i], s=size)
            ax_inset.scatter(mean_err1, mean_err2, label=label, alpha=1, color=cmaps[i], s=size)

    ax.plot([0, 1.2], [0, 1.2], color='black')
    ax.plot([1.0, 1.0], [0, 1.2], color='black', linestyle='dotted')
    ax.plot([0.0, 1.2], [1.0, 1.0], color='black', linestyle='dotted')
    ax.set_xlim(0, 1.2) #  if any_nan_x else 1.0
    ax.set_ylim(0, 1.2) #  if any_nan_y else 1.0
    xticks = ax.get_xticks()
    ax.set_xticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}' 
        for label in xticks])
    yticks = ax.get_yticks()
    ax.set_yticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}' 
        for label in yticks])
    if insert_inset:
        ax_inset.set_xlim(min_x, max_x if not any_nan_x else 1.2)
        ax_inset.set_ylim(min_y, max_y if not any_nan_y else 1.2)
        xticks = ax_inset.get_xticks()
        ax_inset.set_xticklabels([
            'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}' 
            for label in xticks])
        yticks = ax_inset.get_yticks()
        ax_inset.set_yticklabels([
            'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}' 
            for label in yticks])
        ax_inset.plot([1.0, 1.0], [0, 1.2], color='black', linestyle='dotted')
        # Add background to inset
        renderer = fig.canvas.get_renderer()
        coords = ax.transAxes.inverted().transform(ax_inset.get_tightbbox(renderer))
        border = 0.02
        w, h = coords[1] - coords[0] + 2*border
        ax.add_patch(plt.Rectangle(coords[0] - border, w, h, fc="white",
                                transform=ax.transAxes, zorder=2, ec="red", linewidth=2))
        ax.add_patch(plt.Rectangle(
            np.asarray([min_x, min_y]) - border, 
            max_x - min_x + 2 * border, 
            max_y - min_y + 2 * border,
            facecolor='none',
            ec="red", linewidth=2))
        ax.plot([min_x - border, coords[0][0] + border], [max_y + border, coords[0][1] + border], color='red', linestyle='dashed')
        ax.plot([max_x + border, 1.17], [max_y + border, coords[0][1] + border], color='red', linestyle='dashed')
        

    ax.set_xlabel(f'{model_names[0]}')
    ax.set_ylabel(f'{model_names[1]}')
    ax.set_title(title)
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to + '.png')

    return fig

def plot_loss(data, parameter, title, save_to=None):
    parameter_values = parameter['values']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = yaml.safe_load(open('style.yaml'))
    for model in data:
        losses = model['losses']
        mean_loss = np.nan_to_num(np.mean(losses, axis=1), nan=0.5)
        std_loss = np.nan_to_num(np.std(losses, axis=1), nan=0.0)

        ax.plot(parameter_values, mean_loss,
                color=colors[model['method']]['dark'], label=model['method'])

        if (std_loss > 0).any():
            ax.fill_between(
                parameter_values, mean_loss - std_loss, mean_loss + std_loss,
                color=colors[model['method']]['light'],
                alpha=0.5)
    ax.set_xlabel(parameter["name"])
    yticks = ax.get_yticks()
    ax.set_yticklabels([
        'NaN' if np.isclose(label, 1.2, atol=0.1) else f'{label:.1f}' 
        for label in yticks])
    ax.legend()
    fig.tight_layout()
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to + '.png')

    return fig


