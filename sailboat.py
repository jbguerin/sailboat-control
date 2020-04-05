import matplotlib
matplotlib.use('TkAgg')

from numpy import pi, cos, sin, tan, arctan, arctan2, array, sign
from matplotlib.pyplot import *
from numpy.linalg import det, norm
from scipy.linalg import norm


def init_figure(xmin,xmax,ymin,ymax):
    fig = figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    ax.xmin=xmin
    ax.xmax=xmax
    ax.ymin=ymin
    ax.ymax=ymax
    clear(ax)
    return ax


def clear(ax):
    pause(0.001)
    cla()


def angle(x):
    x=x.flatten()
    return arctan2(x[1],x[0])


def plot2D(M, col='black', w=1):
    plot(M[0, :], M[1, :], col, linewidth=w)


def draw_arrow(x, y, θ, L, col):
    e = 0.2
    M1 = L * array([[0, 1, 1 - e, 1, 1 - e], [0, 0, -e, 0, e]])
    M = np.append(M1, [[1, 1, 1, 1, 1]], axis=0)
    R = array([[cos(θ), -sin(θ), x], [sin(θ), cos(θ), y], [0, 0, 1]])
    plot2D(R @ M, col)


def draw_sailboat(x, δs, δr, ψ, awind):
    x = x.flatten()
    θ = x[2]
    hull = array([[-1, 5, 7, 7, 5, -1, -1, -1], [-2, -2, -1, 1, 2, 2, -2, -2], [1, 1, 1, 1, 1, 1, 1, 1]])
    sail = array([[-7, 0], [0, 0], [1, 1]])
    rudder = array([[-1, 1], [0, 0], [1, 1]])
    R = array([[cos(θ), -sin(θ), x[0]], [sin(θ), cos(θ), x[1]], [0, 0, 1]])
    Rs = array([[cos(δs), -sin(δs), 3], [sin(δs), cos(δs), 0], [0, 0, 1]])
    Rr = array([[cos(δr), -sin(δr), -1], [sin(δr), cos(δr), 0], [0, 0, 1]])
    plot2D(R @ hull, 'black')
    plot2D(R @ Rs @ sail, 'red')
    plot2D(R @ Rr @ rudder, 'red')
    #draw_arrow(x[0], x[1], ψ, 5 * awind, 'red')


def update_ax(x, ax, commande):
    abs = str(x[0,0]/100)
    ord = str(x[1,0]/100)

    if abs[0]=="-":
        abs = abs[0:2]
    else:
        abs = abs[0:1]
    if ord[0]=="-":
        ord = ord[0:2]
    else:
        ord = ord[0:1]
    abs = int(abs) * 100
    ord = int(ord) * 100

    limxb, limxh, limyb, limyh = abs-100, abs+100, ord-100, ord+100

    if commande == 1:
        dir = "right"
    elif commande == -1:
        dir = "left"
    else:
        dir = "forward"

    ax.set_xlim(limxb, limxh)
    ax.set_ylim(limyb, limyh)
    ax.text(68 + (limxb + limxh)/2, 53 + (limyb+limyh)/2, "Wind")
    ax.text(50 + (limxb + limxh)/2, 0 + (limyb+limyh)/2, "Direction : "+dir)
    draw_arrow(75+ (limxb + limxh)/2, 40 + (limyb+limyh)/2, ψ, 5 * awind, 'red')


def f(x,u):
    x,u=x.flatten(),u.flatten()
    θ=x[2]; v=x[3]; w=x[4]; δr=u[0]; δsmax=u[1][0];
    w_ap = array([[awind*cos(ψ-θ) - v],[awind*sin(ψ-θ)]])
    ψ_ap = angle(w_ap)
    a_ap=norm(w_ap)
    sigma = cos(ψ_ap) + cos(δsmax)
    if sigma < 0 :
        δs = pi + ψ_ap
    else :
        δs = -sign(sin(ψ_ap))*δsmax
    fr = p4*v*sin(δr)
    fs = p3*a_ap* sin(δs - ψ_ap)
    dx=v*cos(θ) + p0*awind*cos(ψ)
    dy=v*sin(θ) + p0*awind*sin(ψ)
    dv=(fs*sin(δs)-fr*sin(δr)-p1*v**2)/p8
    dw=(fs*(p5-p6*cos(δs)) - p7*fr*cos(δr) - p2*w*v)/p9
    xdot=array([[dx],[dy],[w],[dv],[dw]])
    return xdot,δs


def control(x, a, b, commande):
    # Follow line
    r = 10
    theta = array([x[2]+commande])
    m = array([x[0], x[1]])
    comp1, comp2 = (b-a)/norm(b-a), m-a
    e = det([comp1.flatten(), comp2.flatten()])
    phi = arctan2(b[1]-a[1], b[0]-a[0])
    thetabar = phi-arctan(e/r)
    deltar = (2/pi)*arctan((tan(0.5*(theta-thetabar))))[0][0]

    # Set sail
    deltasmax = pi/4*(cos(ψ-thetabar)+1)

    return array([[deltar],[deltasmax]])  # return array([[0.1],[1]])


p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = 0.1, 1, 6000, 1000, 2000, 1, 1, 2, 300, 10000

dt = 0.1
awind, ψ = 5, pi/2


if __name__ == "__main__":
    x = array([[0, 0, -3, 3, 0]]).T  # x=(x,y,θ,v,w)
    listex, listey = [], []

    ax = init_figure(-100, 100, -60, 60)

    while True:
        clear(ax)

        a = array([[x[0][0]], [x[1][0]]])
        b = array([[x[0][0] + cos(x[2][0])],
                   [x[1][0] + sin(x[2][0])]])
        commande = 1

        listex.append(a[0, 0]), listex.append(b[0, 0])
        listey.append(a[1, 0]), listey.append(b[1, 0])
        # plot([a[0,0],b[0,0]],[a[1,0],b[1,0]],'blue')  # afficher le segment courant
        plot(listex, listey, 'blue')  # afficher la trace complète au fur et à mesure (tous les segments)

        u = control(x, a, b, commande)
        # u1 = angle de la barre
        # u2 = angle de la voile
        xdot, δs = f(x, u)
        x = x + dt * xdot
        draw_sailboat(x, δs, u[0, 0], ψ, awind)
        draw_arrow(75, 40, ψ, 2 * awind, 'red')
        ax.text(68, 53, "Wind")
