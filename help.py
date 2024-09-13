import matplotlib.pyplot as plt
import networkx as nx

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
nodes = ["资源组", "商务组", "律师组"]
edges = [("资源组", "商务组", "提供线索"),
         ("商务组", "律师组", "成单"),
         ("律师组", "商务组", "案件处理"),
         ("律师组", "资源组", "法律问题反馈"),
         ("商务组", "资源组", "线索质量和咨询问题反馈"),
         ("商务组", "律师组", "案件进展通报"),
         ("资源组", "资源组", "线索督查"),
         ("资源组", "资源组", "直播和短视频制作"),
         ("资源组", "资源组", "线下协同")]

# 添加节点
for node in nodes:
    G.add_node(node)

# 添加边
for edge in edges:
    G.add_edge(edge[0], edge[1], label=edge[2])

# 设置图形布局
pos = nx.spring_layout(G)

# 画节点
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')

# 画边
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)

# 画标签
nx.draw_networkx_labels(G, pos, font_size=12)

# 画边标签
edge_labels = {(edge[0], edge[1]): edge[2] for edge in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# 显示图形
plt.title("律师事务所工作流程图")
plt.savefig("./help.jpg")

