import plotly.graph_objects as go
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt

def pair_to_arr(p):
    return p.birth(), p.death(), p.dim()


def process_pairs(ps, remove_zeros=True):
    a = []
    for p in ps:
        a.append([*pair_to_arr(p)])
    a = np.array(a)
    lens = np.abs(a[:,1] - a[:,0])
    if remove_zeros:
        a = a[lens > 0]
    return a


def essential_pair_filter(a, tub=np.inf):
    """
    tub - upper bound on what will be displayed
    """
    return np.max(np.abs(a), axis=1) >= tub


def non_essential_pair_filter(a, tub=np.inf):
    """
    tub - upper bound on what will be displayed
    """
    lens = np.abs(a[:,1] - a[:,0])
    f = lens < tub
    return f


def PD_uncertainty(ps, uncertainty_length = 0.5, remove_zeros=True, show_legend=True, tmax=0.0, tmin=0.0, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    a = process_pairs(ps, remove_zeros)
    dims = np.array(a[:,2], dtype=int) # homology dimension
    cs=plt.get_cmap('Set1')(dims) # set colors

    issublevel = a[0,0] < a[0,1]

    eps = essential_pair_filter(a)
    tmax = np.max((tmax, np.ma.masked_invalid(a[:,:2]).max()))
    tmin = np.min((tmin, np.ma.masked_invalid(a[:,:2]).min()))
    if tmax == tmin:
        # handle identical tmax, tmin
        tmax = tmax*1.1 + 1.0
    span = tmax - tmin
    inf_to = tmax + span/10
    minf_to = tmin - span/10

    # set axes
    xbnds = (minf_to -span/20, inf_to+span/20)
    ybnds = (minf_to-span/20, inf_to+span/20)
    ax.set_xlim(xbnds)
    ax.set_ylim(ybnds)
    ax.set_aspect('equal')

    # add visual lines
    ax.plot(xbnds, ybnds, '--k')
    x_array = np.linspace(*xbnds, 10)
    ax.fill_between(x_array, x_array, x_array + uncertainty_length, alpha=0.2)
    
    if issublevel:
        # +infinity
        ax.plot([minf_to-span/20,tmax], [tmax, tmax], '--r')
    else:
        # -infinity
        ax.plot([tmin,xbnds[1]], [tmin,tmin], '--r')


    # add labels
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    # loop over dimensions
    maxdim = np.max(dims)
    for d in range(maxdim+1):
        dinds = dims == d
        ad = a[dinds]
        cd = cs[dinds]

        ax.scatter(np.NaN, np.NaN, color=plt.get_cmap('Set1')(d), marker='o', label="H{}".format(d))

        neps = non_essential_pair_filter(ad)
        # plot non-essential pairs
        ax.scatter(ad[neps, 0], ad[neps, 1], c=cd[neps], marker='o')

        # plot essential pairs
        eps = essential_pair_filter(ad)
        eb = ad[eps, 0]
        ed = []
        if issublevel:
            ed = [inf_to for _ in eb]
        else:
            ed = [minf_to for _ in eb]
        ax.scatter(eb, ed, c=cd[eps], marker='*')


    if show_legend:
        if issublevel:
            ax.legend(loc='lower right')
        else:
            ax.legend(loc='upper left')

    return fig, ax


def plotly_3D_scatter(X, y, save_path = None, **kwargs):
    # Create a 3D scatter plot
    fig = px.scatter_3d(None, x=X[:,0], y=X[:,1], z=X[:,2], color=y, **kwargs)

    # fig = px.scatter_3d(None, x=X[:,0], y=X[:,1], z=X[:,2])

    # Update chart looks
    fig.update_layout(#title_text="Swiss Roll",
                      showlegend=False,
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.1),
                                            eye=dict(x=1.25, y=1.5, z=1)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))

    # Update marker size
    fig.update_traces(marker=dict(size=3, 
                                  line=dict(color='black', width=0.1)))

    fig.update(layout_coloraxis_showscale=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    if save_path:
        # fig.write_image("images/01/sample_0.pdf")
        fig.write_image(save_path)
    return fig

def plotly_2D_scatter(X_trans, y, save_path = None, **kwargs):
    '''
    Draw dimension reduction results
    
    Input
    ------
    X_trans: datasets after dimension reduction in R^2
    y: color/label
    
    '''

    # Create a scatter plot
    fig = px.scatter(None, x=X_trans[:,0], y=X_trans[:,1], opacity=1, color=y)
    fig.update(layout_coloraxis_showscale=False) # remove color bar
    
    # Change chart background color
    fig.update_layout(dict(plot_bgcolor = 'white'))

    # Update axes lines
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    leave_out = 0.5
    x_low_limit, y_low_limit = np.min(X_trans, axis = 0)
    x_high_limit, y_high_limit = np.max(X_trans, axis = 0)
    fig.update_xaxes(range=[x_low_limit-leave_out, x_high_limit + leave_out])
    fig.update_yaxes(range=[y_low_limit-leave_out, y_high_limit + leave_out])

    # Set figure title
#     fig.update_layout(title_text= title_txt)

    # Update marker size
    fig.update_traces(marker=dict(size=5,
                                 line=dict(color='black', width=0.2)))
    
    fig.update_layout(**kwargs)
    
    if save_path:
        fig.write_image(save_path)

    return fig

def plot_H1(X, F, R, pair, D, save_path = None, thresh=None, **kwargs):
    """
    Plot H1 represnetative on 2D scatter plot

    plot representative
    X: 2-dimensional locations of points
    F: bats FilteredSimplicialComplex
    R: bats ReducedFilteredChainComplex
    pair: bats PersistencePair
    D: N x N distance matrix
    thresh: threshold parameter
    kwargs: passed onto figure layout
    """
    
    if X.shape[1] != 2:
        raise ValueError("Column dimension of X should be 2")
        
    if thresh is None:
        thresh = pair.birth()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1],
        mode='markers',
        showlegend = False,
    ))

    edge_x = []
    edge_y = []
    N = X.shape[0]
    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                edge_x.extend([X[i,0], X[j,0], None])
                edge_y.extend([X[i,1], X[j,1], None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        showlegend = False,
        mode='lines')
     )

    edge_x = []
    edge_y = []
    r = R.representative(pair)
    nzind = r.nzinds()
    cpx = F.complex()
    for k in nzind:
        [i, j] = cpx.get_simplex(1, k)
        if D[i, j] <= thresh:
            edge_x.extend([X[i,0], X[j,0], None])
            edge_y.extend([X[i,1], X[j,1], None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='red'),
        hoverinfo='none',
        name = 'H1',
        mode='lines')
     )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update(layout_coloraxis_showscale=False)
    
    fig.update_layout(**kwargs)

    if save_path:
        fig.write_image(save_path)
    return fig