import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__=='__main__': 
    NUM_L = 12 #the number of the used loudspeakers
    r = np.zeros((NUM_L,3))
    r[:,0] = -2
    if int((NUM_L/2)) != NUM_L/2:
        print('The number of the used loudspeakers is supposed to be even on this code.')
        sys.exit()
    r[:,2] = np.array([-0.2,0.2]*int((NUM_L/2))) 
    r[:,1] = np.linspace(-2.4,2.4,NUM_L)
    Rint = np.array([0.5]) 
    r_c = np.array([[0,0,0]]) #the center of target sphere
    r_s = np.array([-3,0,0]) #the desired position of speaker
    is_silent = np.array([0])
    gamma = np.array([1.0]) #wight of each target sphere
    omega = 2*np.pi*125
    c = 343.0
    N = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r[:,0], r[:,1], r[:,2], c='red', label='position of loudspeakers')
    ax.scatter(r_s[0],r_s[1],r_s[2], c='blue', label='desired microphone')

    r = Rint[0] # 半径を指定
    theta_1_0 = np.linspace(0, np.pi, 200) # θ_1は[0,π/2]の値をとる
    theta_2_0 = np.linspace(-np.pi, np.pi, 400) # θ_2は[0,π/2]の値をとる
    theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0) # ２次元配列に変換
    x = np.cos(theta_2)*np.sin(theta_1) * r # xの極座標表示
    y = np.sin(theta_2)*np.sin(theta_1) * r # yの極座標表示
    z = np.cos(theta_1) * r # zの極座標表示
    ax.plot_surface(x+1,y+1,z,alpha=0.2,)#label='target sphere')
    ax.plot_surface(x+1,y-1,z,alpha=0.2,color='green')
    
    
    #ax.set_title('Simulation condition')
    ax.set_zlim3d(-2,2)
    ax.set_xlim3d(-3,3)
    ax.set_ylim3d(-3,3)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    plt.legend()
    plt.show()