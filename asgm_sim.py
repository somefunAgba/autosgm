import sys, os
from pathlib import Path

import subprocess

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor

# before going to the project root
SVDIR = str(Path(sys.path[0]).parents[0]) + f"/opts/assets/asgm_sim/"
print(SVDIR)
os.makedirs(SVDIR, exist_ok=True)

# act as if we at the project root
sys.path[:0] = [str(Path(sys.path[0]).parents[0])]
# print(sys.path[0])

from cmlibs import *

from asgm import *


import matplotlib.colors as mcolors
# Sort colors by hue, saturation, value and name.
colors, sort_colors = mcolors.TABLEAU_COLORS, False
if sort_colors is True:
    cnames = sorted(
        colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
else:
    cnames = list(colors)
ccolors = [colors[name] for name in cnames]
# ccolors =  ['tab:red', 'tab:blue', 'tab:green', 'tab:olive', 'tab:brown', 'tab:gray', 'tab:cyan', 'tab:purple', 'tab:orange', 'tab:pink', 'black' ]


# ---------------------------------------------

def echvfmt(ax, csts, xlabel, ylabel, grid=True, 
        xm=0, rot=(0,90), xlim_dat=None, ylim_dat=None, extras=False, logy=False, logx=False):

    ax.xaxis.set_tick_params(which='major', labelsize=csts['Fx']-0.5,length=0.5, width=0.4*csts['LW'],pad=0.5)
    ax.xaxis.set_tick_params(which='minor', labelsize=csts['Fx']-0.5,length=0.2, width=0.4*csts['LW'],pad=0.5)   
    ax.set_xlabel(xlabel, fontsize=0.5*csts['Fx'], labelpad=0.5, rotation=rot[0])

    ax.yaxis.set_tick_params(which='major', labelsize=csts['Fy']-0.5,length=0.5, width=0.4*csts['LW'],pad=0.5)    
    ax.yaxis.set_tick_params(which='minor', labelsize=csts['Fy']-0.5,length=0.2, width=0.4*csts['LW'],pad=0.5)   
    lpad = 1
    if rot[1] == 0: lpad = 0.5
    ax.set_ylabel(ylabel, fontsize=0.5*csts['Fy'], labelpad=1, rotation=rot[1])

    TOL= 0 if logy else  0

    if xlim_dat is not None:
        xl, xr = xlim_dat[0], xlim_dat[1]

    if extras:
        ax.yaxis.set_major_formatter(ticker.EngFormatter())    
    
    if logy: 
        ax.set_yscale('log')
        minloc = ticker.LogLocator(subs='auto', numticks=10)
        ax.yaxis.set_minor_locator(minloc)
        ax.yaxis.set_major_formatter(logpow_tick_values)

    if logy:
        # support 'symlog' to include 0, or boolean True for classic log
        if isinstance(logy, str) and logy.lower() == 'symlog':
            # linear region around zero preserves the zero tick while log-scaling large values
            ax.set_yscale('symlog', linthresh=1e-2)
            ax.yaxis.set_major_formatter(logpow_tick_values)
            ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto', numticks=10))
        else:
            ax.set_yscale('log')
            minloc = ticker.LogLocator(subs='auto', numticks=10)
            ax.yaxis.set_minor_locator(minloc)
            ax.yaxis.set_major_formatter(logpow_tick_values)

    if logx:
        ax.set_xscale('log')
        minloc = ticker.LogLocator(subs='auto', numticks=10)
        majloc= ticker.LogLocator(base=10, numticks=50)
        ax.xaxis.set_minor_locator(minloc)
        ax.xaxis.set_major_locator(majloc)
        ax.xaxis.set_major_formatter(logpow_tick_values)

    if ylim_dat is not None:
        yl, yr = ylim_dat[0], ylim_dat[1]
        if logy or extras:
            yl = max(yl, 1e-6)

        if abs(yl-yr) > 1e-3: 
            ax.set_ylim(bottom=yl, top=yr)
            ax.margins(x=0.2, y=0.1, tight=True)

    if logy == 'symlog': 
        labels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox') for label in ax.get_ymajorticklabels()]
        ax.yaxis.set_ticks(ax.get_yticks(), labels)
        ax.set_yticklabels(labels, )
        pass
    else:
        pass
        # labels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox') for label in ax.get_ymajorticklabels()]
        # print('labels', len(labels))
        # # ax.yaxis.set_ticks(ax.get_yticks(), labels)
        # # ax.set_yticklabels(labels, )


    labels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox')
                for label in ax.get_xmajorticklabels()]
    ax.xaxis.set_ticks(ax.get_xticks(), labels)
    ax.set_xticklabels(labels, )
    ax.set_xlim(left=xl, right=xr)    
    # ax.margins(x=0.2, y=0.1, tight=True)

    # if extras:
    ax.autoscale(enable=True, axis="y")

    # plt.tight_layout(pad=0.01)
    if grid: ax.grid(lw=0.05, axis='both')

def postfmt(axp, yls, yrs):
    metriclbls  = [r'$\|\Delta \mathrm{w}[t]\|_{\infty}$', r'$\|e[t]\|_{\infty}$', r'$\|r[t]\|_{\infty}$', r'$\|\alpha[t]\|_{\infty}$' ]
    xlbl  = r"iteration, $t$"
    for ax, metriclbl, yl, yr in zip(axp.flat, metriclbls, yls, yrs):
        lgd = ax.legend(loc='best', mode="shrink", prop={'size':0.67}, ncols=1, borderaxespad=0., fancybox=False, edgecolor='black', frameon=True, alignment='center', handlelength=1, handletextpad=0.3, columnspacing=0.5, labelspacing=0.8)

        frame = lgd.get_frame()
        frame.set_linewidth(0.4*csts['LW'])  
        if yl == yr: yl = 0

        # format axes
        echvfmt(ax, csts, xlbl, metriclbl, grid=True, xm=0, xlim_dat=(0, T), ylim_dat=(yl-1E-8, yr+1E-8), extras=False, logy=False)

    # fmt ytick-labels
    for ax in axp.flat:
        labels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox') for label in ax.get_ymajorticklabels()]
        ax.yaxis.set_ticks(ax.get_yticks(), labels)


# -------------------------------------------------

from matplotlib.animation import FuncAnimation, PillowWriter

from collections import defaultdict


history = defaultdict(lambda: defaultdict(list))  # history[param][state_key] 

# -----------------------------
# Gradient generators
# -----------------------------
def loglik_loss_with_grad_noise(w, d=1, n=32, noise_std=0.01):

    x = torch.randn(n, d)
    logits = x @ w
    probs = torch.sigmoid(logits)

    y = torch.bernoulli(probs.detach())

    loss = F.binary_cross_entropy(probs, y)

    def inject_noise(grad):
        return grad + torch.randn_like(grad) * noise_std

    w.register_hook(inject_noise)
    return loss

def mse_with_grad_noise(w, d=1, w_star=None, Lambda=None, noise_std=0.0, seed=None):
    """
    PyTorch quadratic (MSE) loss with gradient noise injection and
    optional random positive-definite Hessian / non-zero w_star.

    Loss: 0.5 * (w - w_star)^T Lambda (w - w_star)

    Parameters
    - w: torch.nn.Parameter (or tensor requiring grad)
    - w_star: torch tensor same shape as w (if None, sampled randomly)
    - Lambda: PSD matrix of shape (d,d). If None a random PD matrix is created.
    - noise_std: stddev of additive Gaussian noise applied to grad via hook
    - seed: optional integer seed for reproducibility
    """
    device = w.device
    dtype = w.dtype

    # flatten to vector form
    d = w.numel()

    if seed is not None:
        torch.manual_seed(int(seed))

    if w_star is None:
        # non-zero random target
        w_star = torch.randn_like(w, device=device, dtype=dtype)

    if Lambda is None:
        # build a random positive-definite matrix: Lambda = B^T B + eps I
        B = torch.randn(d, d, device=device, dtype=dtype)
        Lambda = B.t().matmul(B)
        Lambda += torch.eye(d, device=device, dtype=dtype) * 1e-6

    # ensure shapes compatible and use 1-D vectors
    diff = (w - w_star).view(-1)            # shape (d,)
    # quadratic form
    loss = 0.5 * diff @ (Lambda.matmul(diff))

    # inject Gaussian noise into gradient of w (if requested)
    if noise_std is not None and float(noise_std) > 0.0:
        def _inject_noise(grad):
            return grad + torch.randn_like(grad) * float(noise_std)
        # register hook on the tensor so gradient gets noise added when backward runs
        w.register_hook(_inject_noise)

    return loss

loss_func = loglik_loss_with_grad_noise
# loss_func  = mse_with_grad_noise
EPOCHS = 1
SPE = 600
T = EPOCHS * SPE

ddim = 3
param_vec = torch.zeros(ddim, dtype=torch.float32)
# param_vec = torch.zeros(ddim, 1, dtype=torch.float32)
params = [torch.nn.Parameter(data=param_vec)]

LRALG = 0
MU = 1E-3
BETA = 0.9
# GAMMA = BETA/(1 + BETA)
GAMMA = 1 - math.sqrt(2*(1 - BETA))
# GAMMA = 0
ETA = (1 - BETA)/(1 - GAMMA)
RHO = 0.1*ETA
print('ETA:', ETA, 'BETA:', BETA, 'GAMMA:', GAMMA, 'RHO:', RHO)

SGM = AutoSGM(params,
              lr_cfg=(True, MU, LRALG),
              beta_cfg=(0.9999, 0.999, BETA, GAMMA, 0, True),
              rc_cfg=(1, 0, 0, 2, 1, EPOCHS, SPE, 1, 0),
              wd_cfg=(RHO, 1),
              eps_cfg=(1e-10, False),
              debug=True)

def _capt(tensor):
    return tensor.detach().cpu().clone().numpy() if isinstance(tensor, torch.Tensor) else tensor

def _ncapt(tensor):
    return tensor.detach().cpu().norm().numpy().item() if isinstance(tensor, torch.Tensor) else np.norm(tensor)

def _supn(v):
    return np.max(np.abs(v))

def SGM_LTV(hstate, beta, gamma, rho, mu):
    gtm1 = hstate['g[t]'][-2]
    gt = hstate['g[t]'][-1]
    wtm1 = hstate['w[t]'][-3]
    wt = hstate['w[t]'][-2]
    wtp1 = hstate['w[t]'][-1]
    atm1 = hstate['a[t]'][-2]
    dtm1 = hstate['d[t]'][-2]
    at = hstate['a[t]'][-1]
    dt = hstate['d[t]'][-1]

    eta = (1-beta)/(1-gamma)
    wbtm1 = wtm1*dtm1
    wbt = wt*dt
    rt = at/atm1
    lrt = at/dt

    e0 = ((gamma*gtm1) - gt)
    e1 = ((beta*wbtm1) - wbt)
    et = e0 + ((rho/eta)*e1) 
    
    odwtp1 = (wtp1 - wt)
    dwt = hstate['dw[t]'][-1]
    dwtp1 = ((beta*rt)*dwt) + ((eta*lrt)*et)

    hstate['r[t]'].append(rt)
    hstate['lr[t]'].append(lrt)    
    hstate['e[t]'].append(et)
    hstate['dw[t]'].append(dwtp1)
    hstate['dwi[t]'].append(odwtp1)
    
    hstate['v_r[t]'].append(_supn(hstate['r[t]']))
    hstate['v_lr[t]'].append(_supn(hstate['lr[t]']))
    hstate['v_dwi[t]'].append(_supn(hstate['dwi[t]']))
    hstate['v_e[t]'].append(_supn(hstate['e[t]']))
    hstate['v_dw[t]'].append(_supn(hstate['dw[t]']))
    hstate['v_w[t]'].append(_supn(hstate['w[t]']))

    bibo = (hstate['v_e[t]'][-1])/(1-gamma)
    sup_dwtp1 = hstate['v_lr[t]'][-1]*(hstate['v_dwi[t]'][0] + bibo)
    hstate['bibo_dw[t]'].append(sup_dwtp1)


# initialize optimizer state
SGM.init() 
for group in SGM.param_groups:
    for p in group['params']:
        states = SGM.state[p]
        history[p]['g[t]'].append(_capt(states['g[t]']))
        history[p]['w[t]'].append(0*_capt(states['w[t]']))  
        history[p]['w[t]'].append(_capt(states['w[t]']))  
        history[p]['a[t]'].append(_capt(states['a[t]']))       
        history[p]['d[t]'].append(_capt(states['d[t]']))     
        history[p]['r[t]'] = []
        history[p]['dwi[t]'].append(_capt(p))
        history[p]['dw[t]'].append(_capt(p))


'''SIMULATION LOOP'''
for t in range(T):
    '''ONE STEP OF SGM'''

    '''SGM PARAMETER UPDATE'''
    SGM.zero_grad()
    loss = loss_func(params[0],  d=ddim, noise_std=0.01)
    loss.backward()
    SGM.step()

    '''RECORD SGM STATES AFTER UPDATE'''
    for group in SGM.param_groups:
        for p in group['params']:
            states = SGM.state[p]
            history[p]['g[t]'].append(_capt(states['g[t]']))
            history[p]['w[t]'].append(_capt(states['w']))  
            history[p]['a[t]'].append(_capt(states['a[t]']))    
            history[p]['d[t]'].append(_capt(states['d[t]']))
            
            SGM_LTV(history[p], beta=group['beta_cfg'][2], gamma=group['beta_cfg'][3], rho=group['wd_cfg'][0], mu=group['lr_cfg'][1])




'''ANIMATION OF SGM TRAJECTORIES'''
def animateX(p, anim_interval=5, use_mp4=True):
    # convert series to numpy arrays once (fast indexed slices)
    y00 = np.asarray(history[p]['v_dwi[t]'])
    y01 = np.asarray(history[p]['v_e[t]'])
    y10 = np.asarray(history[p]['v_r[t]'])
    y11 = np.asarray(history[p]['v_lr[t]'])

    y00i = np.asarray(history[p]['dwi[t]'])
    y01i = np.asarray(history[p]['e[t]'])
    y10i = np.asarray(history[p]['r[t]'])
    y11i = np.asarray(history[p]['lr[t]'])


    # 
    yls = [min(history[p]['v_dwi[t]']), min(history[p]['v_e[t]']), min(history[p]['v_r[t]']), 0]
    yrs = [max(history[p]['v_dwi[t]']), max(history[p]['v_e[t]']), max(history[p]['v_r[t]']), max(history[p]['v_lr[t]'])]

    tmax = max(len(y00), len(y01), len(y10), len(y11))
    x_full = np.arange(tmax)

    # decimate frames so we render at most `max_frames` frames
    max_frames = 300
    step = max(1, int(np.ceil(tmax / max_frames)))
    frames_idx = np.arange(0, tmax, step)

    out_dpi = 1200
    figp, axp = plt.subplots(2,2, figsize=figsz, dpi=out_dpi, tight_layout=True)  # much lower dpi
    (ax00, ax01), (ax10, ax11) = axp
    for ax in axp.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.4*csts['LW'])

    # draw faint static background trace once (cheap)
    ax00.plot(x_full, y00, linewidth=1*csts['LW'],  color='1', alpha=1)
    ax01.plot(x_full, y01,  linewidth=1*csts['LW'], color='1',  alpha=0)
    ax10.plot(x_full, y10, linewidth=1*csts['LW'], color='1', alpha=0)
    ax11.plot(x_full, y11, linewidth=1*csts['LW'], color='1', alpha=0)

    # initialize marker-only lines (no connecting line)
    marker_style = dict(markersize=0.25, markeredgewidth = 0.001)

    l00, = ax00.plot([], [], label=fntscalerl + r'$\|\Delta \mathrm{w}[t]\|_{\infty}$',  lw=0.7*csts['LW'], color='tab:blue', marker='.', **marker_style)
    l01, = ax01.plot([], [], label=fntscalerl + r'$\|e[t]\|_{\infty}$', lw=0.7*csts['LW'], color='tab:orange', marker='.', **marker_style)
    l10, = ax10.plot([], [], label=fntscalerl + r'$\sup_t r[t]$', lw=0.7*csts['LW'], color='tab:green', marker='.', **marker_style)
    l11, = ax11.plot([], [], label=fntscalerl + r'$\sup_t \alpha[t]$', lw=0.7*csts['LW'], color='tab:purple', marker='.', **marker_style)
    #
    lines = [l00, l01, l10, l11]

    n, d = y00i.shape
    print(n, d)
    l00is, l01is, l10is, l11is = [], [], [], []
    for i in range(d):
        ax00.plot(y00i[:, i], linewidth=1*csts['LW'],  color='1', alpha=1)
        ax01.plot(y01i[:, i], linewidth=1*csts['LW'],  color='1', alpha=1)
        ax10.plot(y10i[:, i], linewidth=1*csts['LW'],  color='1', alpha=1)
        ax11.plot(y11i[:, i], linewidth=1*csts['LW'],  color='1', alpha=1)

        tol00, = ax00.plot([], label='_no_legend_',  lw=0.7*csts['LW'], marker='.', **marker_style)
        tol01, = ax01.plot([], label='_no_legend_',  lw=0.7*csts['LW'], marker='.', **marker_style)        
        tol10, = ax10.plot([], label='_no_legend_',  lw=0.7*csts['LW'], marker='.', **marker_style)
        tol11, = ax11.plot([], label='_no_legend_',  lw=0.7*csts['LW'], marker='.', **marker_style)
        l00is.append(tol00)
        l01is.append(tol01)
        l10is.append(tol10)
        l11is.append(tol11)
    
    
    postfmt(axp, yls, yrs)

    def init_func():
        for ln in lines:
            ln.set_data([], [])
        return lines + l00is + l01is + l10is + l11is

    def upd_func(t_idx):
        # t_idx is actual timestep index in frames_idx
        t = int(t_idx)
        # slice safely using pre-made numpy arrays
        ys00 = y00[:t+1] if len(y00) else np.array([])
        ys01 = y01[:t+1] if len(y01) else np.array([])
        ys10 = y10[:t+1] if len(y10) else np.array([])
        ys11 = y11[:t+1] if len(y11) else np.array([])

        ys00i = y00i[:t+1] if len(y00) else np.array([])
        ys01i = y01i[:t+1] if len(y01) else np.array([])
        ys10i = y10i[:t+1] if len(y10) else np.array([])
        ys11i = y11i[:t+1] if len(y11) else np.array([])

        l00.set_data(x_full[:len(ys00)], ys00)
        l01.set_data(x_full[:len(ys01)], ys01)
        l10.set_data(x_full[:len(ys10)], ys10)
        l11.set_data(x_full[:len(ys11)], ys11)

        for i in range(d):
            l00is[i].set_data(x_full[:len(ys00)], ys00i[:, i])
            l01is[i].set_data(x_full[:len(ys01)], ys01i[:, i])
            l10is[i].set_data(x_full[:len(ys10)], ys10i[:, i])
            l11is[i].set_data(x_full[:len(ys11)], ys11i[:, i])

        return lines + l00is + l01is + l10is + l11is

    # animate over decimated frame indices
    animf = FuncAnimation(figp, upd_func, frames=frames_idx, init_func=init_func, interval=anim_interval, repeat=False, blit=True, save_count=len(frames_idx))
    plt.tight_layout(pad=0.1)

    out_mp4 = SVDIR + f'sgmtrajs_{LRALG}.mp4'
    if use_mp4:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=25, codec='libx264')
        print('saving animation to', out_mp4)
        animf.save(out_mp4, writer=writer, dpi=out_dpi)

        out_gif = SVDIR + f'sgmtrajs_{LRALG}_mp4.gif'
        pal = SVDIR + 'palette.png'
        out_dpi = 1200
        cmd1 = ['ffmpeg', '-y', '-i', out_mp4, '-vf',f'fps=25,scale={out_dpi}:-1:flags=lanczos,palettegen', pal]
        cmd2 = ['ffmpeg', '-y', '-i', out_mp4, '-i', pal, '-filter_complex',
        f'fps=25,scale={out_dpi}:-1:flags=lanczos,paletteuse', out_gif]
    
        print('converting mp4 -> gif (palette) to', out_gif)
        subprocess.run(cmd1, check=True)
        subprocess.run(cmd2, check=True)
        try: os.remove(pal)
        except: pass
    else:
        # fallback to Pillow GIF with reasonable dpi and lower fps
        out_gif = SVDIR + f'sgmtrajs_{LRALG}.gif'
        print('ffmpeg not available, saving gif to', out_gif)
        animf.save(out_gif, writer=PillowWriter(fps=25), dpi=out_dpi, savefig_kwargs={'facecolor':'white'})

    plt.close(figp)


'''PLOTTING SGM TRAJECTORIES'''
for group in SGM.param_groups:
    for p in group['params']:

        rlim = 1 if group['beta_cfg'][2] == 0 else 1/group['beta_cfg'][2]

        fntscaler = r'\fontsize{1.5}{1.5}\selectfont '
        fntscaleri = r'\fontsize{0.7}{1}\selectfont '
        fntscalerl = r'\fontsize{1.5}{1.5}\selectfont '
        fntscalerl2 = r'\fontsize{0.5}{1}\selectfont '
        
        csts = {'BM':0.5,'LW':0.15, 'BW':0.15, 'TL':0.92, 'Fy':4, 'Fx':4, 'figsvdir':'','fignm':''}
        figsz = ((10/3)*0.5, 1)
        figp, axp = plt.subplots(2,2, figsize=figsz, dpi=1900, tight_layout=True)
        
        # unpack
        (ax00, ax01), (ax10, ax11) = axp
        for ax in axp.flat:
            for spine in ax.spines.values():
                spine.set_linewidth(0.4*csts['LW']) 

        # limits
        yls = [min(history[p]['v_dwi[t]']), min(history[p]['v_e[t]']), min(history[p]['v_r[t]']), 0]
        yrs = [max(history[p]['v_dwi[t]']), max(history[p]['v_e[t]']), max(history[p]['v_r[t]']), max(history[p]['v_lr[t]'])]
        # print(yls, yrs)

        # visualize in-out trajectories
        # ax00.plot(history[p]['v_dw[t]'], label=fntscalerl + r'$\|\Delta \mathrm{w}[t]\|_{\infty}$', linewidth=1*csts['LW'])
        ax00.plot(history[p]['v_dwi[t]'], label=fntscalerl + r'$\|\Delta \mathrm{w}[t]\|_{\infty}$', linewidth=0.4*csts['LW'])
        # ax00.plot(history[p]['dw[t]'], label='_nolegend_', linewidth=1*csts['LW'])
        ax00.plot(history[p]['dwi[t]'], label='_nolegend_', linewidth=1*csts['LW'])
        # ax00.plot(history[p]['bibo_dw[t]'], label='_nolegend_', linewidth=1*csts['LW'], ls='--')

        ax01.plot(history[p]['v_e[t]'], label=fntscalerl + r'$\|e[t]\|_{\infty}$', linewidth=1*csts['LW'])
        ax01.plot(history[p]['e[t]'], label='_nolegend_', linewidth=1*csts['LW'])

        ax10.plot(history[p]['v_r[t]'], label=fntscalerl + r'$\sup_t r[t]$', linewidth=1*csts['LW'])
        ax10.plot(history[p]['r[t]'], label='_nolegend_', linewidth=1*csts['LW'])
        ax10.axhline(rlim, label='_nolegend_', linewidth=1*csts['LW'], ls='--')

        ax11.plot(history[p]['v_lr[t]'], label=fntscalerl + r'$\sup_t \alpha[t]$', linewidth=1*csts['LW'])
        ax11.plot(history[p]['lr[t]'], label='_nolegend_', linewidth=1*csts['LW'])


        metriclbls  = [r'$\|\Delta \mathrm{w}[t]\|_{\infty}$', r'$\|e[t]\|_{\infty}$', r'$\|r[t]\|_{\infty}$', r'$\|\alpha[t]\|_{\infty}$' ]
        xlbl  = r"iteration, $t$"
        for ax, metriclbl, yl, yr in zip(axp.flat, metriclbls, yls, yrs):
            lgd = ax.legend(loc='best', mode="shrink", prop={'size':0.67}, ncols=1, borderaxespad=0., fancybox=False, edgecolor='black', frameon=True, alignment='center', handlelength=1, handletextpad=0.3, columnspacing=0.5, labelspacing=0.8)

            frame = lgd.get_frame()
            frame.set_linewidth(0.4*csts['LW'])  
            if yl == yr: yl = 0

            # format axes
            echvfmt(ax, csts, xlbl, metriclbl, grid=True, xm=0, xlim_dat=(0, T), ylim_dat=(yl, yr), extras=False, logy=False)

        # fmt ytick-labels
        for ax in axp.flat:
            labels = [fntscaler + label.get_text().replace(r'\mathdefault', r'\hbox') for label in ax.get_ymajorticklabels()]
            ax.yaxis.set_ticks(ax.get_yticks(), labels)

        plt.tight_layout(pad=0.1)
        print('saving figure to ', SVDIR+F'sgmtrajs_{LRALG}.png')

        figp.savefig(SVDIR+f'sgmtrajs_{LRALG}.png', dpi=3600, bbox_inches='tight', pad_inches=0)   
        plt.close(figp)

        # animation
        # animateX(p)





