from jax import numpy as jnp

from seaborn import sns


colors = sns.color_palette("colorblind")


def pred_3D_test(solver, aligned_long, preds, 
            fig = None, ax = None, z_downsample = 1,
            min_t = 0, max_t = 601, skip = 100, 
            extr_cmap = sns.dark_palette(colors[3], as_cmap=True),
            cobras_cmap = sns.light_palette(colors[-1], as_cmap=True,),
            extr_kws = {},
            cobras_kws = {},
            q_thresh = 0.2,
            view_kwargs = dict(elev=40., azim=-55),
            min_x = 0, max_x = 256 * jnp.pi,
            x_ticks = None,
            x_bounds_nudge = 10,
            t_bounds_nudge = 10,
            wire_lw = 1
            ):
    
    cobras_kwargs = dict(vmin = 2.5, vmax = 3.5, alpha = 0.5, zorder =0, s = 1)
    extr_kwargs = dict(s = 50, vmin = 0.15, vmax = 0.2, alpha = 1.0, 
                       zorder=2, depthshade=False, edgecolor = 'k', lw=0.5)
    
    if cobras_kws:
        cobras_kwargs.update(cobras_kws)
    if extr_kws:
        extr_kwargs.update(extr_kws)
        
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10,20),facecolor='white')
        ax = fig.add_subplot(111,projection ='3d', computed_zorder=False)

    # x_idx = jnp.arange(0,len(solver.x))
    x_mask_idx = jnp.logical_and(solver.x >= min_x, solver.x <= max_x)
    x_mask_idx = jnp.arange(0,len(solver.x))[x_mask_idx] 
    
    t_idx = jnp.arange(min_t,  max_t)[::skip]
    x = solver.x[x_mask_idx]
    print(x[0], x[-1])
    X_mesh, T_mesh = jnp.meshgrid(x[::z_downsample], jnp.arange(min_t, max_t))

    
    ax.scatter(X_mesh, T_mesh, 
                jnp.zeros_like(X_mesh),
                c = preds[min_t:max_t,x_mask_idx], 
                cmap = cobras_cmap, **cobras_kwargs,
                )

    for ti in t_idx:
        qoi_plot = jnp.abs(aligned_long[ti])[x_mask_idx]
        q_idx = qoi_plot > q_thresh
        ax.plot(x, jnp.ones_like(x)*ti, qoi_plot,
                c = 'k', lw =wire_lw, zorder= extr_kwargs['zorder']-0.1)
        
        # restrict to only plotting the extreme values
        ax.scatter(x[q_idx], jnp.ones_like(x[q_idx])*ti, qoi_plot[q_idx],
                c = qoi_plot[q_idx], 
                cmap = extr_cmap, 
                **extr_kwargs)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_zticks([0.1, 0.2, 0.3])

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    else:
        ax.set_xticks(jnp.arange(0,9) *32* jnp.pi, 
                    labels = [f'{32*i}' + r'$\pi$' for i in range(9)])

    # Change gridline style 
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]: 
        axis._axinfo['grid']['linestyle'] = '--' 
        axis._axinfo['grid']['linewidth'] = 0.8 
        axis._axinfo['grid']['color'] = 'gray'
        axis._axinfo['axisline']['linewidth'] = 1.2
        axis._axinfo['axisline']['color'] = 'black'
        axis.pane.set_edgecolor('black')
        axis.pane.set_linewidth(1.2)
        
    ax.xaxis._axinfo['grid']['linewidth'] = 0
    ax.yaxis._axinfo['grid']['linewidth'] = 0

    ax.set_box_aspect(aspect=(12,12,2))
    ax.set_xlim(min_x-x_bounds_nudge, max_x+x_bounds_nudge)
    ax.set_ylim(min_t-t_bounds_nudge, max_t)
    ax.set_zlim(0, 0.3)

    for i, y in enumerate(ax.get_yticks()[1:-1]):
        # L = 256 * jnp.pi
        ax.plot([min_x, max_x], [y, y], [0, 0], color='gray', linestyle='--', linewidth=0.8)
        

    ax.view_init(**view_kwargs)
    # Get limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    # Nudge the line inward so it's guaranteed to be visible
    eps_x = 1.5e-3 * (xmax - xmin)
    eps_y = 1.5e-3 * (ymax - ymin)

    ax.plot(
        [xmin + eps_x, xmin + eps_x],
        [ymin + eps_y, ymin + eps_y],
        [zmin, zmax],
        color='black',
        linewidth=1.2
    )
    return ax, fig