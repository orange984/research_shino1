import numpy as np


def main():
  data = np.loadtxt("/home/subaru/workspace/research_shino1/NEW_py_ver/TSP/POMO/result/20220226_111222_test__tsp_n50/norm_dists")
  print(np.shape(data))
  #print(np.mean(data.min(axis=1)))
  #print(np.shape(data.min(axis=1)))
  new_data = []
  val = 0
  i = 0
  while i < 200000:
    val = np.mean(data[i:(i+10000)].min(axis=1))
    new_data.append(val)
    i += 10000

  np.savetxt("/home/subaru/workspace/research_shino1/NEW_py_ver/TSP/POMO/result/20220226_111222_test__tsp_n50/new_graph_data",new_data)


if __name__ == "__main__":
    main()
