import numpy as np
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
from pathlib import Path

from . import loadingbar


def plot_spins(spins, axes=None,
               resize=True, noticks=True, show=False,
               imshow_kwargs=None):
    """
    Draw a grid of spins to axes

    Uses imshow(). Draws image centred on 0 to fit a 1x1 box unless extent
    passed onto imshow_kwargs. Use resize=False to prevent the axis limits
    from being resized. 

    If using show=True, do not pass any axes.
    """

    # 1. Check and set default parameters

    if axes is None and not show:
        raise ValueError("Must pass axes if not showing.")
    if axes is not None and show:
        raise ValueError("Cannot pass axes if showing.")

    # Left, Right, Bottom, Top
    aspect = spins.shape[0] / spins.shape[1]
    if aspect >= 1:  # wider than tall
        l, r = -0.5, 0.5
        b, t = -0.5 / aspect, 0.5 / aspect
    else:  # taller than wide
        l, r = -0.5 * aspect, 0.5 * aspect
        b, t = -0.5, 0.5
    default_extent = (l, r, b, t)

    if not imshow_kwargs:
        kwa = {}
    else:
        kwa = imshow_kwargs

    kwa.setdefault("extent", default_extent)
    kwa.setdefault("cmap", "binary")
    kwa.setdefault("vmax", +1)
    kwa.setdefault("vmin", -1)

    # 2. Actual plotting

    if show:
        fig = plt.figure(figsize=(6, 6))
        pad = 0.05
        axes = fig.add_axes((pad, pad, 1 - 2 * pad, 1 - 2 * pad))

    # I want the spins to be indexed by [x][y], so I have to do some flipping.
    # Of course it's somewhat pointless since everything is rotation/parity
    # symmetric, but it's slightly more general this way.
    image = axes.imshow(np.flip(spins.T, axis=0), **kwa)

    if noticks:
        axes.set_xticks([])
        axes.set_yticks([])

    if resize:
        axes.set_aspect("equal")

    if show:
        plt.figure(fig.number)
        plt.show()

    return image


def _anim_func_spins(spins, image):
    """func parameter to FuncAnimation() constructor"""

    image.set_data(spins)
    return image,


def animate_spins(simulation, axes, show=False, repeat=1,
                  resize=True, noticks=True, imshow_kwargs=None,
                  anim_kwargs=None):
    """Create a FuncAnimation object based on a particular simulation"""

    if anim_kwargs is None:
        anim_kwargs = {}

    anim_kwargs.setdefault("interval", 50)
    anim_kwargs.setdefault("repeat_delay", 500)

    fig = axes.figure
    iternum, Nx, Ny = simulation.shape
    init_spins = simulation[0]

    image = plot_spins(init_spins, axes, resize, noticks, imshow_kwargs)

    anim = mpl.animation.FuncAnimation(
        fig, _anim_func_spins,
        frames=repeat * tuple(simulation[t] for t in range(iternum)),
        fargs=(image,),
        **anim_kwargs
    )

    if show:

        plt.figure(fig.number)
        plt.show()

    return anim


def _anim_func_mosaic(frame, image_list, text, lbar):

    t, ens_state = frame

    for sys_state, image in zip(ens_state, image_list):
        image.set_data(sys_state)

    if lbar is not None:
        lbar.print_next()

    if text is None:
        return tuple(image_list)
    else:
        text.set_text(str(t))
        return tuple(image_list) + (text,)


def animate_mosaic(ensemble, fig=None, timestamp=False, verbose=False,
                   pad=0.05, bbox=(0, 0, 1, 1),
                   show=False, imshow_kwargs=None, anim_kwargs=None,
                   saveas=None):
    """Draw out a datagen.Ensemble as a pretty mosaic animation"""

    if anim_kwargs is None:
        anim_kwargs = {}

    anim_kwargs.setdefault("interval", 50)
    anim_kwargs.setdefault("repeat_delay", 500)

    sysnum = ensemble.sysnum
    iternum = ensemble.iternum
    N = int(np.ceil(np.sqrt(sysnum)))
    ens_arr = ensemble.asarray()

    if fig is None:
        fig = plt.figure(figsize=(5, 5))

    axes_list = []
    image_list = []

    k = 0

    for i in range(N):
        for j in range(N):

            if k < sysnum:

                # Create grid of axes
                left, bottom, width, height = bbox
                bounds = (
                    (left + (i + pad) / N) * (width - pad / N),
                    (bottom + (j + pad) / N) * (height - pad / N),
                    ((1 - pad) / N) * (width - pad / N),
                    ((1 - pad) / N) * (height - pad / N)
                )
                ax = fig.add_axes(bounds)
                axes_list.append(ax)

                # Plot out initial spins
                init_spins = ens_arr[0, k, ...]
                im = plot_spins(init_spins, axes=ax, resize=False,
                                imshow_kwargs=imshow_kwargs)
                image_list.append(im)

            k += 1

    if timestamp:
        text = axes_list[0].text(0, 0, "", color=(1, 0, 0), weight="bold",
                                 ha="left", va="top")
    else:
        text = None

    if verbose:
        lbar = loadingbar.LoadingBar(iternum)
    else:
        lbar = None

    anim = mpl.animation.FuncAnimation(
        fig, _anim_func_mosaic,
        frames=tuple((t, ens_arr[t, ...]) for t in range(iternum)),
        fargs=(image_list, text, lbar),
        init_func=lambda: 0,
        **anim_kwargs
    )

    if saveas is not None:

        saveas = str(Path(saveas).with_suffix(".mp4"))
        anim.save(saveas)

    if show:

        plt.figure(fig.number)
        plt.show()

    return fig, axes_list, anim
