import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import linalg

from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)

def Triangle_GenerateInternalGridAndInternalCells_Local(n_pt=25):
    """
    生成三角形内结点和内网格（画图用）
    参数：
        n_pt 边上的结点数，注意 n_pt-1 为2、3、4、6 的倍数
    """

    n_node = (n_pt+1)*n_pt//2     # 全部内结点数（含边角）
    n_tria = (n_pt-1)*(n_pt-1)    # 小三角形数
    n_edge = 3*n_pt - 2         # 单元边界点数，形成闭路多一个

    gridCrds_local = np.zeros(shape=(n_node, 3), dtype=float)
    gridCell_array = np.zeros(shape=(n_tria, 3), dtype=int)
    eleEdges_array = np.zeros(n_edge, dtype=int)

    i_node = 0
    for l1 in np.arange(n_pt,0,-1)-1:
        for l2 in np.arange(n_pt-l1,0,-1)-1:
            l3 = n_pt - l1 - l2 -1
            gridCrds_local[i_node, :] = np.array([l1, l2, l3], dtype=float)/(n_pt-1)
            i_node += 1

    i_tri  = 0
    start2 = 0
    for i_row in range(n_pt-1):
        start1 = start2
        start2 += i_row + 1
        for idx in range(i_row + 1):
            i1 = idx + start1
            i2 = idx + start2
            i3 = i2 + 1
            gridCell_array[i_tri, :] = np.array([i1, i2, i3])
            i_tri += 1
        if i_row <= n_pt - 2:
            for idx in range(i_row):
                i1 = idx + start1
                i2 = idx + start2 + 1
                i3 = i1 + 1
                gridCell_array[i_tri, :] = np.array([i1, i2, i3])
                i_tri += 1

    beg_edge, end_edge = 0, -1
    beg_idx,  end_idx  = 0,  0
    for i_row in range(n_pt-1):
        beg_idx += i_row
        end_idx  = beg_idx + i_row
        eleEdges_array[beg_edge] = beg_idx
        eleEdges_array[end_edge] = end_idx
        beg_edge += 1
        end_edge -= 1
    for idx in range(n_pt):
        eleEdges_array[beg_edge+idx] = end_idx + 1 + idx

    return (gridCrds_local, gridCell_array, eleEdges_array)

# 生成内网格局部坐标
def Quadrilateral_GenerateInternalGridAndInternalCells_Local(n_pt=25):
    """
    功能：生成四边形内结点和内网格（画图用）
    参数：n_pt 为边上的结点数
    """
    xi_array = np.linspace(-1, 1, n_pt)
    eta_array = np.linspace(-1, 1, n_pt)

    return (xi_array, eta_array)



class Element2D():
    """
    平面单元基类,派生类命名 Name + cCsSiI(c个角结点、s个边结点、i个内结点)
    """
    class_dict = dict()
    config_dict = dict()

    # 初始化函数
    def __init__(self, node_crds_global, ele_id, node_ids):
        self.node_crds_global  = node_crds_global[node_ids]  # 结点全局坐标
        self.ele_id    = ele_id           # 单元编号
        self.node_ids   = node_ids         # 单元结点编号列表

        config = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]
        
        if geom == "Triangle":
            self.Triangle_GenerateInternalGrid_Global()
        elif geom == "Quadrilateral":
            self.Quadrilateral_GenerateInternalGrid_Global()
#
    def GenerateInternalGrid_Global(self): # 生成三角形内网格点的全局坐标
        config = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]
        
        if geom == "Triangle":
            self.Triangle_GenerateInternalGrid_Global()
        elif geom == "Quadrilateral":
            self.Quadrilateral_GenerateInternalGrid_Global()

    # 生成内全部网格点的全局坐标
    def Triangle_GenerateInternalGrid_Global(self): # 生成三角形内网格点的全局坐标
        config = Element2D.config_dict[self.__class__.__name__]

        gridCrds_local    = config["gridCrds_local"]
        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        node_crds_global       = self.node_crds_global
        n_internalNode    = len(gridCrds_local)

        shape_function_values = np.empty(shape=(n_internalNode, num_node))

        # 待修改
        for i_pt in range(n_internalNode):
            l1, l2, l3 = tuple(gridCrds_local[i_pt])
            for i_nd in range(num_node):
                shape_function_values[i_pt, i_nd] = shape_function_list[i_nd](l1, l2, l3)

        self.gridCrds_global = shape_function_values.dot(self.node_crds_global)

#
    def Quadrilateral_GenerateInternalGrid_Global(self): # 生成四边形内网格点的全局坐标
        config = Element2D.config_dict[self.__class__.__name__]

        xi_array        = config["xi_array"]
        eta_array       = config["eta_array"]
        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        node_crds_global  = self.node_crds_global
        n_pt          = config["num_internalGrid"]
        dim_global      = config["dim_global"] # 2

        self.gridCrds_global = np.zeros(shape=(n_pt, n_pt, dim_global), dtype= float)

        for i_pt in range(n_pt):
            xi = xi_array[i_pt]
            for j_pt in range(n_pt):
                eta = eta_array[j_pt]
                for i_nd in range(num_node):
                    shape_function_val = shape_function_list[i_nd](xi, eta)
                    self.gridCrds_global[i_pt, j_pt, 0] += shape_function_val * node_crds_global[i_nd, 0]
                    self.gridCrds_global[i_pt, j_pt, 1] += shape_function_val * node_crds_global[i_nd, 1]

    # 计算并返回单元内多点处的全局坐标及场函数值
    def CalculateSiteFunction_Points(self, node_phi_array, point_crds_array):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        x_crds = self.node_crds_global[:,0]
        y_crds = self.node_crds_global[:,1]

        num_point = len(point_crds_array)
        phi_array = np.zeros(num_point, dtype=float)
        x_array  = np.zeros(num_point, dtype=float) 
        y_array  = np.zeros(num_point, dtype=float)

        for i_point in range(num_point):
            point_crds = point_crds_array[i_point, :]
            for i_node in range(num_node):
                shape_function = shape_function_list[i_node]
                phi_array[i_point] += shape_function(*point_crds) * node_phi_array[i_node]
                x_array[i_point]  += shape_function(*point_crds) * x_crds      [i_node]
                x_array[i_point]  += shape_function(*point_crds) * y_crds      [i_node]

        return x_array, y_array, phi_array

    # 计算并返回单元内多点处的全局坐标及场函数值
    def Triangle_CalculateSiteFunction_Points(self, node_phi_array, point_crds_array):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        x_crds = self.node_crds_global[:,0]
        y_crds = self.node_crds_global[:,1]

        num_point = len(point_crds_array)
        phi_array = np.zeros(num_point, dtype=float)
        x_array  = np.zeros(num_point, dtype=float) 
        y_array  = np.zeros(num_point, dtype=float)

        for i_point in range(num_point):
            point_crds = point_crds_array[i_point, :]
            for i_node in range(num_node):
                shape_function = shape_function_list[i_node]
                phi_array[i_point] += shape_function(*point_crds) * node_phi_array[i_node]
                x_array[i_point]  += shape_function(*point_crds) * x_crds      [i_node]
                x_array[i_point]  += shape_function(*point_crds) * y_crds      [i_node]

        return x_array, y_array, phi_array

    # 本函数可以和三角形的函数全并，计算并返回单元内多点处的全局坐标及场函数值
    def Quadrilateral_CalculateSiteFunction_Points(self, node_phi_array, point_crds_array):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        x_crds = self.node_crds_global[:,0]
        y_crds = self.node_crds_global[:,1]

        num_point = len(point_crds_array)
        phi_array = np.zeros(num_point, dtype=float)
        x_array  = np.zeros(num_point, dtype=float) 
        y_array  = np.zeros(num_point, dtype=float)

        for i_point in range(num_point):
            point_crds = point_crds_array[i_point, :]
            for i_node in range(num_node):
                shape_function_val = shape_function_list[i_node](*point_crds)
                phi_array[i_point] += shape_function_val * node_phi_array[i_node]
                x_array[i_point]  += shape_function_val * x_crds      [i_node]
                y_array[i_point]  += shape_function_val * y_crds      [i_node]

        return x_array, y_array, phi_array
#
    # 计算并返回全部内网格点处的网格点全局坐标及场函数值
    def CalculateSiteFunction_Grid(self, node_phi_array):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_CalculateSiteFunction_Grid(node_phi_array)
        elif geom == "Quadrilateral":
            self.Quadrilateral_CalculateSiteFunction_Grid(node_phi_array)

    # 计算并返回全部内网格点处的网格点全局坐标及场函数值
    def Triangle_CalculateSiteFunction_Grid(self, node_phi_array):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        gridCrds_local    = config["gridCrds_local"]
        shape_function_list = config["shape_function_list"]

        x_grid_array = self.gridCrds_global[:,0]
        y_grid_array = self.gridCrds_global[:,1]

        phi_grid_array = np.zeros_like(x_grid_array)
        for i in range(len(x_grid_array)):
            l1, l2, l3 = tuple(gridCrds_local[i,:])
            for i_node in range(num_node):
                phi_grid_array[i] += shape_function_list[i_node](l1, l2, l3)*node_phi_array[i_node]

        return x_grid_array, y_grid_array, phi_grid_array

    # 计算并返回全部内网格点处的网格点全局坐标及场函数值
    def Quadrilateral_CalculateSiteFunction_Grid(self, node_phi_array):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        n_pt          = config["num_internalGrid"]

        xi_array        = config["xi_array"]
        eta_array       = config["eta_array"]
        shape_function_list = config["shape_function_list"]

        x_grid_array = self.gridCrds_global[:,:,0]
        y_grid_array = self.gridCrds_global[:,:,1]

        phi_grid_array = np.zeros(shape=(n_pt, n_pt), dtype=float)

        for i_pt in range(n_pt):
            xi = xi_array[i_pt]
            for j_pt in range(n_pt):
                eta = eta_array[j_pt]
                for i_nd in range(num_node):
                    shape_function_val = shape_function_list[i_nd](xi, eta)
                    phi_grid_array[i_pt, j_pt] += shape_function_val * node_phi_array[i_nd]

        return x_grid_array, y_grid_array, phi_grid_array

    def CalculateCenterCrds(self):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_CalculateCenterCrds()
        elif geom == "Quadrilateral":
            self.Quadrilateral_CalculateCenterCrds()
#
    # 计算并返回三角形形心点全局坐标值
    def Triangle_CalculateCenterCrds(self):
        config  = Element2D.config_dict[self.__class__.__name__]

        num_node        = config["num_node"]
        shape_function_list = config["shape_function_list"]

        x_crds = self.node_crds_global[:,0]
        y_crds = self.node_crds_global[:,1]

        xc, yc = 0, 0
        for i_node in range(num_node):
            xc += shape_function_list[i_node](1/3, 1/3, 1/3)*x_crds[i_node]
            yc += shape_function_list[i_node](1/3, 1/3, 1/3)*y_crds[i_node]

        return xc, yc

    # 计算并返回四边形形心点全局坐标值
    def Quadrilateral_CalculateCenterCrds(self):
        config  = Element2D.config_dict[self.__class__.__name__]

        x_crds = self.node_crds_global[:,0]
        y_crds = self.node_crds_global[:,1]

        xc, yc, _ = self.Quadrilateral_CalculateSiteFunction_Points(x_crds, np.array([[0,0]]))

        return xc, yc

    def DrawElementId(self, ax, show_ele_id, projection="2d"):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_DrawElementId(ax, show_ele_id, projection="2d")
        elif geom == "Quadrilateral":
            self.Quadrilateral_DrawElementId(ax, show_ele_id, projection="2d")
#
    # 画单元标识号
    def Triangle_DrawElementId(self, ax, show_ele_id, projection="2d"):
        if not show_ele_id:
            return
        xc, yc = self.Triangle_CalculateCenterCrds()
        if projection=="2d":
            ax.text(xc, yc, r'$\circled{'+str(self.ele_id)+'}$', fontsize=18, horizontalalignment='center', verticalalignment='center')
        else:
            #ax.text(xc, yc, 0, r'$\circled{'+str(self.ele_id)+'}$', (0,0,1), color="black")
            ax.text3D(xc, yc, 0, "E", zdir="z")

    # 画单元标识号
    def Quadrilateral_DrawElementId(self, ax, show_ele_id, projection="2d"):
        if not show_ele_id:
            return
        xc, yc = self.Quadrilateral_CalculateCenterCrds()
        if projection=="2d":
            ax.text(xc, yc, r'$\circled{'+str(self.ele_id)+'}$', fontsize=18, horizontalalignment='center', verticalalignment='center')
        else:
            ax.text(xc, yc, 0, r'$\circled{'+str(self.ele_id)+'}$', (0,0,1), color="black")
#
    def DrawSiteFunction_Global(self, node_phi_array, ax, norm, show_ele_id=False):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_DrawSiteFunction_Global(node_phi_array, ax, norm, show_ele_id=False)
        elif geom == "Quadrilateral":
            self.Quadrilateral_DrawSiteFunction_Global(node_phi_array, ax, norm, show_ele_id=False)

    # 画全局坐标系中的场函数曲面图
    def Triangle_DrawSiteFunction_Global(self, node_phi_array, ax, norm, show_ele_id=False):
        config = Element2D.config_dict[self.__class__.__name__]

        num_node = config["num_node"]
        gridCell_array = config["gridCell_array"]
        eleEdges_array = config["eleEdges_array"]

        xs, ys, phis = self.Triangle_CalculateSiteFunction_Grid(node_phi_array)

        # 画单元场函数曲面图
        ax.plot_trisurf(xs, ys, phis, triangles=gridCell_array, cmap="jet", norm=norm, lightsource=ls, linewidth=0.1, antialiased=True)

        # 画曲面、平面上的单元边界曲线图
        ax.plot3D(xs[eleEdges_array], ys[eleEdges_array], phis[eleEdges_array],  "-k", lw=4)
        ax.plot3D(xs[eleEdges_array], ys[eleEdges_array], phis[eleEdges_array]*0, "-k", lw=2)

        eleNodeCrds_local = config["eleNodeCrds_local"]
        x_nodes, y_nodes, phi_nodes = self.CalculateSiteFunction_Points(node_phi_array, eleNodeCrds_local)

        node_x_array = self.node_crds_global[:,0]
        node_y_array = self.node_crds_global[:,1]

        for i_node in range(num_node):
            x, y, phi = node_x_array[i_node], node_y_array[i_node], node_phi_array[i_node]
            ax.plot3D([x,x], [y,y], [0,phi], ":k")         # 单元结点连线
            ax.plot3D([x,x], [y,y], [0,phi], "ok", lw=4)     # 单元结点连线
        
        self.Triangle_DrawElementId(ax, show_ele_id)

    # 画全局坐标系中的场函数曲面图
    def Quadrilateral_DrawSiteFunction_Global(self, node_phi_array, ax, norm, show_ele_id=False):
        config = Element2D.config_dict[self.__class__.__name__]

        num_node = config["num_node"]

        xs, ys, phis = self.Quadrilateral_CalculateSiteFunction_Grid(node_phi_array)

        # 画单元场函数曲面图
        ax.plot_surface(xs, ys, phis, cmap="jet", norm=norm, lightsource=ls, linewidth=0.1, antialiased=True)

        # 画曲面、平面上的单元边界曲线图
        ax.plot3D(xs[ 0,:], ys[ 0,:], phis[ 0,:],  "-k", lw=4)
        ax.plot3D(xs[ 0,:], ys[ 0,:], phis[ 0,:]*0, "-k", lw=2)
        ax.plot3D(xs[-1,:], ys[-1,:], phis[-1,:],  "-k", lw=4)
        ax.plot3D(xs[-1,:], ys[-1,:], phis[-1,:]*0, "-k", lw=2)

        ax.plot3D(xs[:, 0], ys[:, 0], phis[:, 0],  "-k", lw=4)
        ax.plot3D(xs[:, 0], ys[:, 0], phis[:, 0]*0, "-k", lw=2)
        ax.plot3D(xs[:,-1], ys[:,-1], phis[:,-1],  "-k", lw=4)
        ax.plot3D(xs[:,-1], ys[:,-1], phis[:,-1]*0, "-k", lw=2)

        node_x_array = self.node_crds_global[:,0]
        node_y_array = self.node_crds_global[:,1]

        for i_node in range(num_node):
            x, y, phi = node_x_array[i_node], node_y_array[i_node], node_phi_array[i_node]
            ax.plot3D([x,x], [y,y], [0,phi], ":k")         # 单元结点连线
            ax.plot3D([x,x], [y,y], [0,phi], "ok", lw=4)     # 单元结点连线
        
        self.Quadrilateral_DrawElementId(ax, show_ele_id)

    def DrawElementDomain_Global(self, ax, show_ele_id=True):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_DrawElementDomain_Global(ax, show_ele_id=True)
        elif geom == "Quadrilateral":
            self.Quadrilateral_DrawElementDomain_Global(ax, show_ele_id=True)

    def Triangle_DrawElementDomain_Global(self, ax, show_ele_id=True):
        config  = Element2D.config_dict[self.__class__.__name__]

        eleEdges_array    = config["eleEdges_array"]
        shape_function_list = config["shape_function_list"]

        X = self.gridCrds_global[:,0]
        Y = self.gridCrds_global[:,1]

        ax.plot(X[eleEdges_array],  Y[eleEdges_array],  "-k", lw=2)   # 单元边界
        ax.plot(self.node_crds_global[:,0], self.node_crds_global[:,1], "ok", lw=4)   # 单元结点
        
        self.Triangle_DrawElementId(ax, show_ele_id)


    def Quadrilateral_DrawElementDomain_Global(self, ax, show_ele_id=True):
        X = self.gridCrds_global[:,:,0]
        Y = self.gridCrds_global[:,:,1]

        ax.plot(X[ 0,:], Y[ 0,:], "-k", lw=1)
        ax.plot(X[-1,:], Y[-1,:], "-k", lw=1)

        ax.plot(X[:, 0], Y[:, 0], "-k", lw=1)
        ax.plot(X[:,-1], Y[:,-1], "-k", lw=1)
        
        ax.plot(self.node_crds_global[:,0], self.node_crds_global[:,1], "ok", lw=4)   # 单元结点
        
        self.Quadrilateral_DrawElementId(ax, show_ele_id)

    def DrawShapeFunction_Global(self, which_node, ax, show_ele_id=False):
        config  = Element2D.config_dict[self.__class__.__name__]
        geom = config["geom"]

        if geom == "Triangle":
            self.Triangle_DrawShapeFunction_Global(which_node, ax, show_ele_id=False)
        elif geom == "Quadrilateral":
            self.Quadrilateral_DrawShapeFunction_Global(which_node, ax, show_ele_id=False)

    def Triangle_DrawShapeFunction_Global(self, which_node, ax, show_ele_id=False):
        config = Element2D.config_dict[self.__class__.__name__]

        num_node = config["num_node"]

        node_phi_array = np.zeros(num_node)
        if self.node_ids.count(which_node): # which_node 在 self.node_ids 中
            node_idx = self.node_ids.index(which_node) # 提取结点序号
            node_phi_array[node_idx] = 1.0

        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        self.Triangle_DrawSiteFunction_Global(node_phi_array, ax, norm, show_ele_id)
        self.Triangle_DrawElementId(ax, show_ele_id, projection="3d")

    def Quadrilateral_DrawShapeFunction_Global(self, which_node, ax, show_ele_id=False):
        config = Element2D.config_dict[self.__class__.__name__]

        num_node = config["num_node"]

        node_phi_array = np.zeros(num_node)
        if self.node_ids.count(which_node):           # which_node 在 self.node_ids 中
            node_idx = self.node_ids.index(which_node)  # 提取结点序号
            node_phi_array[node_idx] = 1.0

        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        self.Quadrilateral_DrawSiteFunction_Global(node_phi_array, ax, norm, show_ele_id)
        self.Quadrilateral_DrawElementId(ax, show_ele_id, projection="3d")


def Triangle_Element_Config(num_node, shape_function_list, gridCrds_local, gridCell_array, eleEdges_array, eleNodeCrds_local, n_pt=25 ):  # 边界结点数
    return {
            "geom": "Triangle",       # 几何形状
            "dim_global": 2, "dim_local": 3,
            "num_node": num_node, "eleNodeCrds_local": eleNodeCrds_local, "shape_function_list": shape_function_list,
            "num_internalGrid": n_pt, "gridCrds_local": gridCrds_local,
            "gridCell_array": gridCell_array,
            "eleEdges_array": eleEdges_array
          }

def Create_Triangle_Element_Class(
                    elementClassName,   # 命名约定 Triangle + cCsSiI
                    eleNodeCrds_local,  # 采用面积坐标数组 [[l_11, l_12, l_13], [l_21, l_22, l_23], ..., [l_n1, l_n2, l_n3]] 
                    shape_function_list=None, # 列表 [N_1, N_2, ..., N_n]
                    n_pt = 25
                  ):
    if elementClassName not in Element2D.class_dict.keys():
        num_node = len(shape_function_list)
        row_num, col_num = eleNodeCrds_local.shape

        assert num_node >= 3
        assert num_node == row_num
        assert col_num == 3
        assert not (n_pt-1) % 6

        element_class = type(elementClassName, (Element2D,), {})

        gridCrds_local, gridCell_array, eleEdges_array = Triangle_GenerateInternalGridAndInternalCells_Local(n_pt)
        
        element_config = Triangle_Element_Config(num_node, shape_function_list, gridCrds_local,
                                   gridCell_array, eleEdges_array, eleNodeCrds_local, n_pt)

        Element2D.class_dict [elementClassName] = element_class
        Element2D.config_dict[elementClassName] = element_config

    return Element2D.class_dict[elementClassName]

Triangle3C = Create_Triangle_Element_Class(
                        "Triangle3C",
                         np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float),
                         [lambda l1,l2,l3: l1,lambda l1,l2,l3: l2,lambda l1,l2,l3: l3]
                        )
    
Triangle3C1S = Create_Triangle_Element_Class(
                        "Triangle3C1S",  # 三角形单元 含3个角结点、1个边中结点，1-2边的中点是结点
                         np.array([[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0]],dtype=float),
                         [
                             lambda l1,l2,l3:l1-2*l1*l2,
                             lambda l1,l2,l3:l2-2*l1*l2,
                             lambda l1,l2,l3:l3,
                             lambda l1,l2,l3:4*l1*l2
                         ]
                        )

Triangle3C2S = Create_Triangle_Element_Class(
                        "Triangle3C2S",  # 三角形单元 含3个角结点、2个边中结点，1-2、2-3边的中点是结点
                         np.array([[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0,0.5,0.5]],dtype=float),
                         [
                             lambda l1,l2,l3:l1-2*l1*l2,
                             lambda l1,l2,l3:l2-2*l1*l2-2*l2*l3,
                             lambda l1,l2,l3:l3-2*l2*l3,
                             lambda l1,l2,l3:4*l1*l2,
                             lambda l1,l2,l3:4*l2*l3
                         ]
                        )

Triangle3C3S = Create_Triangle_Element_Class(
                        "Triangle3C3S",  # 三角形单元 含3个角结点、3个边中结点，1-2、2-3、3-1边的中点是结点
                         np.array([[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5]], dtype=float),
                         [
                            lambda l1,l2,l3: (2*l1-1)*l1,
                            lambda l1,l2,l3: (2*l2-1)*l2,
                            lambda l1,l2,l3: (2*l3-1)*l3,
                            lambda l1,l2,l3: 4*l1*l2,
                            lambda l1,l2,l3: 4*l2*l3,
                            lambda l1,l2,l3: 4*l3*l1
                         ] # 形函数
                        )

def Quadrilateral_Element_Config(num_node, shape_function_list, xi_array, eta_array, eleNodeCrds_local, n_pt=25):
    return {
            "geom": "Quadrilateral",       # 几何形状
            "dim_global": 2, "dim_local": 2,
            "num_node": num_node,
            "shape_function_list": shape_function_list,
            "xi_array": xi_array,
            "eta_array": eta_array,
            "eleNodeCrds_local": eleNodeCrds_local,
            "num_internalGrid": n_pt,
            "iso_flag": True
          }

def Create_Quadrilateral_Element_Class(
                    elementClassName,   # 命名约定 Quadrilateral + cCsSiI
                    eleNodeCrds_local,  # 采用局部坐标数组 [[xi_1, eta_1], [xi_2, eta_2], ..., [xi_n, eta_n]] 
                    shape_function_list=None, # 列表 [N_1, N_2, ..., N_n]
                    n_pt = 25
                  ):

    if elementClassName not in Element2D.class_dict.keys():
        num_node = len(shape_function_list)
        row_num, col_num = eleNodeCrds_local.shape

        assert num_node >= 4       # 至少 4 个结点
        assert num_node == row_num   # 结点数匹配
        assert col_num == 2       # 恰好两个局部坐标 xi、eta
        assert not (n_pt-1) % 6    # 单元边界分段数是 6 的倍数

        element_class = type(elementClassName, (Element2D,), {})

        xi_array, eta_array = Quadrilateral_GenerateInternalGridAndInternalCells_Local(n_pt)
        
        element_config = Quadrilateral_Element_Config(num_node, shape_function_list, xi_array, eta_array, eleNodeCrds_local, n_pt)

        Element2D.class_dict [elementClassName] = element_class
        Element2D.config_dict[elementClassName] = element_config

    return Element2D.class_dict[elementClassName]

Quadrilateral4C = Create_Quadrilateral_Element_Class( "Quadrilateral4C",
                                     np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=float),
                                     [
                                        lambda xi,eta: 0.25*(1-xi)*(1-eta),
                                        lambda xi,eta: 0.25*(1+xi)*(1-eta),
                                        lambda xi,eta: 0.25*(1+xi)*(1+eta),
                                        lambda xi,eta: 0.25*(1-xi)*(1+eta)
                                     ]
                                    )

Quadrilateral4C1S = Create_Quadrilateral_Element_Class( "Quadrilateral4C1S",
                                      np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-1]],dtype=float),
                                      [
                                        lambda xi,eta: 0.25*(1-xi)*(1-eta)-0.25*(1-xi*xi)*(1-eta),
                                        lambda xi,eta: 0.25*(1+xi)*(1-eta)-0.25*(1-xi*xi)*(1-eta),
                                        lambda xi,eta: 0.25*(1+xi)*(1+eta),
                                        lambda xi,eta: 0.25*(1-xi)*(1+eta),
                                        lambda xi,eta: 0.5*(1-xi*xi)*(1-eta)
                                      ]
                                    )

Quadrilateral4C4S = Create_Quadrilateral_Element_Class(
                                       "Quadrilateral4C4S",
                                       np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-1],[1,0],[0,1],[-1,0]],dtype=float),
                                       [
                                            lambda xi,eta: 0.25*(1-xi)*(1-eta)-0.25*(1-xi*xi)*(1-eta) -0.25*(1-xi)*(1-eta*eta),
                                            lambda xi,eta: 0.25*(1+xi)*(1-eta)-0.25*(1-xi*xi)*(1-eta) -0.25*(1+xi)*(1-eta*eta),
                                            lambda xi,eta: 0.25*(1+xi)*(1+eta)-0.25*(1+xi)*(1-eta*eta)-0.25*(1-xi*xi)*(1+eta),
                                            lambda xi,eta: 0.25*(1-xi)*(1+eta)-0.25*(1-xi*xi)*(1+eta) -0.25*(1-xi)*(1-eta*eta),
                                            lambda xi,eta: 0.5*(1-xi*xi)*(1-eta),
                                            lambda xi,eta: 0.5*(1+xi)*(1-eta*eta),
                                            lambda xi,eta: 0.5*(1-xi*xi)*(1+eta),
                                            lambda xi,eta: 0.5*(1-xi)*(1-eta*eta)
                                       ]
                                    )



class ElementSystem:
    def __init__(self, node_crds_global_array, ele_info_list):
        self.element_dict = dict()
        self.node_crds_global_array = node_crds_global_array

        for (e_id, node_index_list, e_type) in ele_info_list:
            ele = e_type(node_crds_global_array, e_id, node_index_list)
            self.element_dict[e_id] = ele

    def DrawElementDomain_Global(self, ax=None, figsize=None, show_node_id=False, show_ele_id=False):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

        for (e_id, ele) in self.element_dict.items():
            ele.DrawElementDomain_Global(ax, show_ele_id=True)

        if ax is None: plt.show()

    def DrawShapeFunction_Global(self, node_id, ax=None, figsize=None, show_node_id=False, show_ele_id=False):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca(projection="3d")

        for (e_id, ele) in self.element_dict.items():
            ele.DrawShapeFunction_Global(node_id, ax)

        if ax is None: plt.show()

    def DrawSiteFunction_Global(self, node_phi_array, norm, ax=None, figsize=None, show_node_id=False, show_ele_id=False):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca(projection="3d")

        for (e_id, ele) in self.element_dict.items():
            ele.DrawSiteFunction_Global(node_phi_array[ele.node_ids], ax=ax, norm=norm)

        if ax is None: plt.show()

    def DrawNodeIds_Global(self, node_ids, ax):
        pass