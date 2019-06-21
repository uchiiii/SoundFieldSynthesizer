import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

if __name__=='__main__': 
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
    ax.plot_surface(x,y,z,alpha=0.2,)#label='target sphere')
    ax.plot_surface(x+4,y-4,z,alpha=0.2,color='green')
    
    
    #ax.set_title('Simulation condition')
    ax.set_zlim3d(-3,3)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    plt.legend()