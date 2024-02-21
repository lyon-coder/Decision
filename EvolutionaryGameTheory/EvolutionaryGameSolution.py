import matplotlib.pyplot as plt
from pylab import *
import math
import gc
import imageio
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']


def GeneralTrans(s_e_s, s_e_e, s_a_s, s_a_e, v_e, v_a):
    a = 1.0
    b = 1.0
    t1 = s_e_e / v_e
    t2 = s_a_e / v_a
    k1 = v_e / s_e_s
    k2 = v_a / s_a_s
    return a, b, t1, t2, k1, k2


def HeuristicTrans(s_e_s, s_e_e, s_a_s, s_a_e, v_e, v_a, h_a):
    a = 1.0
    b = 1.0
    t2 = s_a_e / v_a
    k2 = v_a / s_a_s
    if h_a < 0.0:
        if math.pow(v_e, 2) / (2.0 * h_a) < s_e_s:
            return a, b, 1e+8, t2, 1e+8, k2
        if math.pow(v_e, 2) / (2.0 * h_a) < s_e_e:
            ttc = CalEquation(0.5 * h_a, v_e, -s_e_s)
            k1 = 1.0 / ttc
            return a, b, 1e+8, t2, k1, k2
    ttc = CalEquation(0.5 * h_a, v_e, -s_e_s)
    t1 = CalEquation(0.5 * h_a, v_e, -s_e_e)
    k1 = 1.0 / ttc
    return a, b, t1, t2, k1, k2


def CalEquation(a, b, c):
    return (-b + math.sqrt(math.pow(b, 2) - 4.0 * a * c)) / (2.0 * a)


def f(x, y, a, c, t1, t2, k1, k2):
    return x * (1-x) * (-2 * k2 * y - y * a * c * t2 + 2 * a * t2 - y * a * t2)


def g(x, y, a, c, t1, t2, k1, k2):
    return y * (1-y) * (-2 * k1 * x - x * a * c * t1 + 2 * a * t1 - x * a * t1)


def calculateValue(initX, initY, dt, epoch, a, c, t1, t2, k1, k2):
    x = []
    y = []

    x.append(initX)
    y.append(initY)

    for index in range(epoch):
        tempx = x[-1] + (f(x[-1], y[-1], a, c, t1, t2, k1, k2)) * dt
        tempy = y[-1] + (g(x[-1], y[-1], a, c, t1, t2, k1, k2)) * dt

        x.append(tempx)
        y.append(tempy)
    return (x, y)


def PlotEvolution(prob_set, a, c, t1, t2, k1, k2):
    D = []
    for param in prob_set:
        x = param[0]
        y = param[1]
        d = calculateValue(x, y, 0.01, 1000, a, c, t1, t2, k1, k2)
        D.append(d)
    return D


def CalculateSaddle(a, c, t1, t2, k1, k2):
    x = 2.0*a*t1 / (2.0*k1 + a*c*t1 + a*t1)
    y = 2.0*a*t2 / (2.0*k2 + a*c*t2 + a*t2)
    return x, y

def UniformMotion(state, delta_t):
    s_ego_s = state[0] - state[4]*delta_t
    s_ego_e = state[1] - state[4]*delta_t
    s_agent_s = state[2] - state[5]*delta_t
    s_agent_e = state[3] - state[5]*delta_t
    return (s_ego_s, s_ego_e, s_agent_s, s_agent_e, state[4], state[5])


def Display(state, prob_set):
    s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent = state
    a, c, t1, t2, k1, k2 = GeneralTrans(s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent
                                        )
    D = PlotEvolution(prob_set, a, c, t1, t2, k1, k2)
    plt.figure(1)
    plt.plot(D[0][0], D[0][1], label='Equal Right of Way:x=0.5,y=0.5')
    plt.plot(D[1][0], D[1][1], label='Self Driving Priority:x=0.6,y=0.3')
    plt.plot(D[2][0], D[2][1], label='Self Driving Priority:x=0.7,y=0.1')
    plt.plot(D[3][0], D[3][1], label='Other Driving Priority:x=0.1,y=0.6')
    plt.plot(D[4][0], D[4][1], label='Other Driving Priority:x=0.15,y=0.7')
    plt.plot(D[5][0], D[5][1], label='Other Driving Priority:x=0.45,y=0.6')

    plt.ylabel("$y$", fontsize=18)
    plt.xlabel("$x$", fontsize=18)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.legend()

    plt.figure(2)
    t_set = [x * 0.01 for x in range(500)]
    plt.plot(t_set, (D[0][0])[:500], label='x=0.5,y=0.5')
    plt.plot(t_set, (D[1][0])[:500], label='x=0.6,y=0.3')
    plt.plot(t_set, (D[2][0])[:500], label='x=0.7,y=0.1')
    plt.plot(t_set, (D[3][0])[:500], label='x=0.1,y=0.3')
    plt.plot(t_set, (D[4][0])[:500], label='x=0.7,y=0.9')
    plt.plot(t_set, (D[5][0])[:500], label='x=0.2,y=0.8')

    plt.ylabel("$x$", fontsize=18)
    plt.xlabel("$t$", fontsize=18)
    plt.legend()

    plt.figure(3)
    t_set = [x * 0.01 for x in range(500)]
    plt.plot(t_set, (D[0][1])[:500], label='x=0.5,y=0.5')
    plt.plot(t_set, (D[1][1])[:500], label='x=0.6,y=0.3')
    plt.plot(t_set, (D[2][1])[:500], label='x=0.7,y=0.1')
    plt.plot(t_set, (D[3][1])[:500], label='x=0.1,y=0.3')
    plt.plot(t_set, (D[4][1])[:500], label='x=0.7,y=0.9')
    plt.plot(t_set, (D[5][1])[:500], label='x=0.2,y=0.8')

    plt.ylabel("$y$", fontsize=18)
    plt.xlabel("$t$", fontsize=18)
    plt.legend()

    prob_set = [[0.5, 0.5]]
    h_a_set = [0.5, 1.0, 1.5]
    D = []
    D_H = []
    for h_a in h_a_set:
        a, c, t1, t2, k1, k2 = HeuristicTrans(s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent, h_a
                                              )
        D = PlotEvolution(prob_set, a, c, t1, t2, k1, k2)
        D_H.append(D[0])
    # print(D_H)
    plt.figure(4)
    t_set = [x * 0.01 for x in range(500)]
    plt.plot(t_set, (D_H[0][0])[:500], label='heuristics a=0.5')
    plt.plot(t_set, (D_H[1][0])[:500], label='heuristics a=1.0')
    plt.plot(t_set, (D_H[2][0])[:500], label='heuristics a=1.5')
    # plt.plot(t_set, (D_H[3][0])[:500], label='heuristics a=-0.5')
    # plt.plot(t_set, (D_H[4][0])[:500], label='heuristics a=-1.0')
    # plt.plot(t_set, (D_H[5][0])[:500], label='heuristics a=-1.5')

    plt.ylabel("$x$", fontsize=18)
    plt.xlabel("$t$", fontsize=18)
    plt.legend()

    plt.figure(5)
    t_set = [x * 0.01 for x in range(500)]
    plt.plot(t_set, (D_H[0][1])[:500], label='heuristics a=0.5')
    plt.plot(t_set, (D_H[1][1])[:500], label='heuristics a=1.0')
    plt.plot(t_set, (D_H[2][1])[:500], label='heuristics a=1.5')
    # plt.plot(t_set, (D_H[3][1])[:500], label='heuristics a=-0.5')
    # plt.plot(t_set, (D_H[4][1])[:500], label='heuristics a=-1.0')
    # plt.plot(t_set, (D_H[5][1])[:500], label='heuristics a=-1.5')

    plt.ylabel("$y$", fontsize=18)
    plt.xlabel("$t$", fontsize=18)
    plt.legend()

    plt.show()

def ConsecutiveDisplay(state, prob, outpath):
    s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent = state
    a, c, t1, t2, k1, k2 = GeneralTrans(s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent
                                        )
    D = PlotEvolution(prob, a, c, t1, t2, k1, k2)
    saddle_x, saddle_y = CalculateSaddle(a, c, t1, t2, k1, k2)
    fig = plt.figure(dpi=300)
    plt.plot(saddle_x, saddle_y, 'ro')

    plt.plot([saddle_x, 0], [saddle_y, 0], color='black')
    plt.plot([saddle_x, 1], [saddle_y, 0], color='black')
    plt.plot([saddle_x, 0], [saddle_y, 1], color='black')
    plt.plot([saddle_x, 1], [saddle_y, 1], color='black')
    plt.plot([0, 1, 1, 0, 0], [1, 1, 0, 0, 1], color='black')

    plt.plot(D[0][0], D[0][1], label='Equal Right of Way:x=0.5,y=0.5')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylabel("$y$", fontsize=18)
    plt.xlabel("$x$", fontsize=18)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.legend()

    plt.savefig("{}".format(outpath))
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)
    plt.close('all')
    gc.collect()


def DetTrjDisplay(det_as, det_bs, det_cs, det_ds, det_es, tr_as, tr_bs, tr_cs, tr_ds, tr_es, delta_t, outpath):
    ts = [x * delta_t for x in range(len(det_as))]
    fig = plt.figure(dpi=300)
    plt.plot(ts, det_as, label='Det A = (0, 1)')
    plt.plot(ts, tr_as, label='Trj A = (0, 1)')
    plt.plot(ts, det_bs, label='Det B = (0, 0)')
    plt.plot(ts, tr_bs, label='Trj B = (0, 0)')
    plt.plot(ts, det_cs, label='Det C = (1, 0)')
    plt.plot(ts, tr_cs, label='Trj C = (1, 0)')
    plt.plot(ts, det_ds, label='Det D = (1, 1)')
    plt.plot(ts, tr_ds, label='Trj D = (1, 1)')
    plt.plot(ts, det_es, label='Det E = (saddle x, saddle y)')
    plt.plot(ts, tr_es, label='Trj E = (saddle x, saddle y)')
    plt.ylabel("$value$", fontsize=14)
    plt.xlabel("$time(s)$", fontsize=14)
    plt.legend()

    plt.savefig("{}".format(outpath))
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)
    plt.close('all')
    gc.collect()


def CreateGIF(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.2)
    return


def process_record(state):
    a, c, t1, t2, k1, k2 = GeneralTrans(state[0], state[1], state[2], state[3], state[4], state[5]
                                        )
    saddle_x = 2.0*a*t1 / (2.0*k1 + a*c*t1 + a*t1)
    saddle_y = 2.0*a*t2 / (2.0*k2 + a*c*t2 + a*t2)
    det_A = np.sign(2.0*a*t1*(2.0*k2+a*c*t2-a*t2))
    tr_A = np.sign(-2.0*k2-a*c*t2+a*t2-2.0*a*t1)
    det_B = np.sign(4.0*a*a*t1*t2)
    tr_B = np.sign(2.0*a*t2 + 2.0*a*t1)
    det_C = np.sign(2.0*a*t2*(2.0*k1+a*c*t1-a*t1))
    tr_C = np.sign(-2.0*a*t2-2.0*k1-a*c*t1+a*t1)
    det_D = np.sign((-2.0*k2-a*c*t2+a*t2)*(-2.0*k1-a*c*t1+a*t1))
    tr_D = np.sign(2.0*(k1+k2)+a*c*(t1+t2)-a*(t1+t2))
    det_E = np.sign(-4.0*a*a*t1*t2*(2.0*k1+a*c*t1-a*t1)*(2.0*k2+a*c*t2-a*t2)/(2.0*k1+a*c*t1+a*t1)/(2.0*k2+a*c*t2+a*t2))
    print(-4.0*a*a*t1*t2*(2.0*k1+a*c*t1-a*t1)*(2.0*k2+a*c*t2-a*t2)/(2.0*k1+a*c*t1+a*t1)/(2.0*k2+a*c*t2+a*t2))
    print("aaaaaaaaaa: ", k1, t1, k2, t2, (2.0*k1+a*c*t1-a*t1), (2.0*k2+a*c*t2-a*t2))
    tr_E = 0
    return (saddle_x, saddle_y), det_A, tr_A, det_B, tr_B, det_C, tr_C, det_D, tr_D, det_E, tr_E


if __name__ == '__main__':
    state = (24.0, 20.0, 23.0, 20.5, 3.0, 2.0) # s_ego_s, s_ego_e, s_agent_s, s_agent_e, v_ego, v_agent

    prob_set = [[0.5, 0.5], [0.6, 0.3], [0.7, 0.1],
                [0.1, 0.6], [0.15, 0.7], [0.45, 0.6]]
    prob = [[0.5, 0.5]]
    # Display(state, prob_set)
    fig_index = 0
    out_path_list = list()
    det_as = list() 
    det_bs = list()
    det_cs = list()
    det_ds = list()
    det_es = list()
    tr_as = list()
    tr_bs = list()
    tr_cs = list()
    tr_ds = list()
    tr_es = list()
    while state[1] > 0.0 and state[3] > 0.0:
        print(state[1], state[3])
        outpath = f'D:\gif\c_display\{fig_index}.jpg'
        ConsecutiveDisplay(state, prob, outpath)
        out_path_list.append(outpath)
        (saddle_x, saddle_y), det_a, tr_a, det_b, tr_b, det_c, tr_c, det_d, tr_d, det_e, tr_e = process_record(state)
        det_as.append(det_a)
        det_bs.append(det_b)
        det_cs.append(det_c)
        det_ds.append(det_d)
        det_es.append(det_e)
        tr_as.append(tr_a)
        tr_bs.append(tr_b)
        tr_cs.append(tr_c)
        tr_ds.append(tr_d)
        tr_es.append(tr_e)
        state = UniformMotion(state, 0.1)
        fig_index += 1
    gif_name = f'D:\gif\c_display\ConsecutiveDecision.gif'
    CreateGIF(out_path_list, gif_name)
    records_path = f'D:\gif\detr/records.jpg'
    DetTrjDisplay(det_as, det_bs, det_cs, det_ds, det_es, tr_as, tr_bs, tr_cs, tr_ds, tr_es, 0.1, records_path)



