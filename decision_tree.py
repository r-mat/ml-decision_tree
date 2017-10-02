import numpy as np
import matplotlib.pyplot as plt
import pydotplus
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import tree
from sklearn import __version__ as sklearn_version
if sklearn_version < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

def main():
    #データセットのロード
    iris_data = np.loadtxt("data/iris.data.txt",delimiter=",",usecols=(0,1,2,3))
    iris_target = np.loadtxt("data/iris.data.txt",delimiter=",",usecols=(4), dtype=np.dtype("U16"))
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_names = np.array(['setosa', 'versicolor', 'virginica'])
    X = np.array(iris_data[:,[0,2]])  # sepal length and petal length
    y = []
    for i in range(len(iris_target)):
        if(iris_target[i] == 'Iris-setosa'):
            y.append(0)
        elif(iris_target[i] == 'Iris-versicolor'):
            y.append(1)
        elif(iris_target[i] == 'Iris-virginica'):
            y.append(2)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    max_depth    = None
    random_state = 3

    #トレーニングセットを読み込み、学習
    clf_m = DecisionTree(criterion="gini", max_depth=max_depth, random_state=random_state)
    clf_m.fit(X_train, y_train)
    my_score = clf_m.score(X_test, y_test)

    clf_s = tree.DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=random_state)
    clf_s.fit(X_train, y_train)
    sklearn_score = clf_s.score(X_test ,y_test)
    
    #--- print score
    print("-"*50)
    print("my decision tree score:" + str(my_score))
    print("scikit-learn decision tree score:" + str(sklearn_score))

    #---print feature importances
    print("-"*50)
    f_importance_m = clf_m.feature_importances_
    f_importance_s = clf_s.feature_importances_

    print ("my decision tree feature importances:")
    for f_name, f_importance in zip(np.array(feature_names)[[0,2]], f_importance_m):
        print("    ",f_name,":", f_importance)

    print ("sklearn decision tree feature importances:")
    for f_name, f_importance in zip(np.array(feature_names)[[0,2]], f_importance_s):
        print("    ",f_name,":", f_importance)
        
    #--- output decision region
    plot_result(clf_m, X_train,y_train, X_test, y_test, "my_decision_tree")
    plot_result(clf_s, X_train,y_train, X_test, y_test, "sklearn_decision_tree")
    
    #---output decision tree chart
    tree_ = TreeStructure()
    dot_data_m = tree_.export_graphviz(clf_m.tree, feature_names=np.array(feature_names)[[0,2]], class_names=target_names)
    graph_m = pydotplus.graph_from_dot_data(dot_data_m)

    dot_data_s = tree.export_graphviz(clf_s, out_file=None, feature_names=np.array(feature_names)[[0,2]], class_names=target_names, 
                                      filled=True, rounded=True, special_characters=True)  
    graph_s = pydotplus.graph_from_dot_data(dot_data_s)

    graph_m.write_png("chart_my_decision_tree.png")
    graph_s.write_png("chart_sklearn_decision_tree.png")


    
def plot_result(clf, X_train,y_train, X_test, y_test, png_name):
    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]
    
    markers = ('s','d', 'x','o', '^', 'v')
    colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    labels = ('setosa', 'versicolor', 'virginica')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    dx = 0.02
    X1 = np.arange(x1_min, x1_max, dx)
    X2 = np.arange(x2_min, x2_max, dx)
    X1, X2 = np.meshgrid(X1, X2)
    Z = clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    plt.figure(figsize=(12, 10))
    plt.clf()
    plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=1.0, c=cmap(idx),
                    marker=markers[idx], label=labels[idx])
        
    plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c="", marker="o", s=100,  label="test set")

    plt.title("Decision region(" + png_name + ")")
    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")
    plt.grid()
    #--plt.show()
    plt.savefig("decision_region_" + png_name + ".png", dpi=300)


#ノードを表現するクラス
class Node(object):

    #コンストラクター
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        #評価指標 gini / entropy
        self.criterion    = criterion
        #決定木の最大のノードの深さ
        self.max_depth    = max_depth
        #乱数のシード
        self.random_state = random_state
        #このノードの深さ
        self.depth        = None
        #左の子ノード
        self.left         = None
        #右の子ノード
        self.right        = None
        #分割に利用する特徴量
        self.feature      = None
        #分割に利用する特徴量の判定の閾値
        self.threshold    = None
        #ノードのデータの中で一番多いカテゴリ
        self.label        = None
        #ノードの評価関数
        self.impurity     = None
        #情報利得
        self.info_gain    = None
        #そのノードに属するデータ数
        self.num_samples  = None
        #そのノードに属するデータの目的変数のカテゴリごとのデータ数（配列）
        self.num_classes  = None
    
    #ノードを分割する
    #sample :データ
    #target :目的変数
    #depth :ノードの深さ
    #ini_num_classes :カテゴリ数
    def split_node(self, sample, target, depth, ini_num_classes):
        self.depth = depth

        #ノードのデータ数、カテゴリごとのデータ数を取得
        self.num_samples = len(target)
        self.num_classes = [len(target[target==i]) for i in ini_num_classes]
       
        #カテゴリ数が1つのみである場合、終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return

        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        #カテゴリの中で一番データ数が多いものをlabelに格納
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        #情報利得を格納
        self.impurity = self.criterion_func(target)
        
        #特徴量の種類数を取得
        num_features = sample.shape[1]
        self.info_gain = 0.0

        #乱数のシードがNone=乱数のシード自体をランダムに設定
        if self.random_state!=None:
            np.random.seed(self.random_state)
        

        f_loop_order = np.random.permutation(num_features).tolist()

        #ここから各特徴量ごとにループし、情報利得が最大となる情報量と閾値をもとめる
        #決定木の学習のコアな部分
        for f in f_loop_order:
            uniq_feature = np.unique(sample[:, f])
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0

            for threshold in split_points:
                target_l = target[sample[:, f] <= threshold] 
                target_r = target[sample[:, f] >  threshold]
                val = self.calc_info_gain(target, target_l, target_r)
                #1つ前の情報利得よりも今回の情報利得が大きければ置き換え
                if self.info_gain < val:
                    self.info_gain = val
                    self.feature   = f
                    self.threshold = threshold

        #情報利得が0=最も不純度が小さい場合、終了
        if self.info_gain == 0.0:
            return
        #引数指定した深さがMAXに等しい場合、終了
        if depth == self.max_depth:
            return

        #左子ノードを生成。再帰的に子ノードも分割
        sample_l   = sample[sample[:, self.feature] <= self.threshold]
        target_l   = target[sample[:, self.feature] <= self.threshold]
        self.left  = Node(self.criterion, self.max_depth)
        self.left.split_node(sample_l, target_l, depth + 1, ini_num_classes)

        #右子ノードを生成。再帰的に子ノードも分割
        sample_r   = sample[sample[:, self.feature] > self.threshold]
        target_r   = target[sample[:, self.feature] > self.threshold]
        self.right = Node(self.criterion, self.max_depth)
        self.right.split_node(sample_r, target_r, depth + 1, ini_num_classes)

    #評価関数
    def criterion_func(self, target):
        classes = np.unique(target)
        numdata = len(target)
        #ジニ不純度の場合
        if self.criterion == "gini":
            val = 1
            for c in classes:
                p = float(len(target[target == c])) / numdata
                val -= p ** 2.0
        #情報エントロピーの場合
        elif self.criterion == "entropy":
            val = 0
            for c in classes:
                p = float(len(target[target == c])) / numdata
                if p!=0.0:
                    val -= p * np.log2(p)
        return val

    #ノード分割した場合の情報利得を計算する
    def calc_info_gain(self, target_p, target_cl, target_cr):
        #親ノードの利得
        cri_p  = self.criterion_func(target_p)
        #子左ノードの利得
        cri_cl = self.criterion_func(target_cl)
        #子右ノードの利得
        cri_cr = self.criterion_func(target_cr)
        return cri_p - len(target_cl)/float(len(target_p))*cri_cl - len(target_cr)/float(len(target_p))*cri_cr

    #学習した結果を元に予測する関数
    def predict(self, sample):
        #すでに葉ノードに達している場合はそのノードのラベルを応答
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        #そうでない場合は、子ノードへ制御を渡す。
        #再帰的に計算される
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

#決定木の特徴量の重要度を計算するオブジェクト
class TreeAnalysis(object):
    #コンストラクター
    def __init__(self):
        self.num_features = None
        self.importances  = None

    #そのノード以下の特徴量ごとの重要度=情報利得×データ数を計算し、
    # 内部変数importances配列に保持   
    def compute_feature_importances(self, node):
        if node.feature == None:
            return
        
        self.importances[node.feature] += node.info_gain*node.num_samples
        
        self.compute_feature_importances(node.left)
        self.compute_feature_importances(node.right)
    
    #特徴量の重要度配列を取得
    def get_feature_importances(self, node, num_features, normalize=True):
        self.num_features = num_features
        self.importances  = np.zeros(num_features)
        
        self.compute_feature_importances(node)
        #※規格化
        self.importances /= node.num_samples

        #規格化
        if normalize:
            normalizer = np.sum(self.importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                self.importances /= normalizer
        return self.importances
    
#決定木オブジェクト            
class DecisionTree(object):

    #コンストラクター
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        #ルートノード
        self.tree          = None
        #評価関数
        self.criterion     = criterion
        #決定木のMAX深さ
        self.max_depth     = max_depth
        #乱数のシード
        self.random_state  = random_state
        self.tree_analysis = TreeAnalysis()

    #ルートノードを作成し、学習をする
    def fit(self, sample, target):

        #ルートノードの作成
        self.tree = Node(self.criterion, self.max_depth, self.random_state)
        #学習
        self.tree.split_node(sample, target, 0, np.unique(target))
        #重要度の計算
        self.feature_importances_ = self.tree_analysis.get_feature_importances(self.tree, sample.shape[1])

    #学習した決定木を使って予測をする
    def predict(self, sample):
        pred = []
        for s in sample:
            pred.append(self.tree.predict(s))
        return np.array(pred)

    #予測のスコア（予測値=正解の割合）を算出
    def score(self, sample, target):
        return sum(self.predict(sample) == target)/float(len(target))
    
    
#チャート図作成クラス
class TreeStructure(object):
    def __init__(self):
        self.num_node = None
        self.dot_data = None
        
    def print_tree(self, node, feature_names, class_names, parent_node_num):
        node.my_node_num = self.num_node
        node.parent_node_num = parent_node_num

        tree_str = ""
        #葉ノードの場合
        if node.feature == None or node.depth == node.max_depth:
            tree_str += str(self.num_node) + " [label=<" + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + class_names[node.label] + ">, fillcolor=\"#00000000\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> " 
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str
        #葉ノードではない場合
        else:
            tree_str += str(self.num_node) + " [label=<" + feature_names[node.feature] + " &le; " + str(node.threshold) + "<br/>" \
                                           + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + class_names[node.label] + ">, fillcolor=\"#00000000\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> " 
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str

            #子ノードについてチャート図作成
            self.num_node+=1
            self.print_tree(node.left, feature_names, class_names, node.my_node_num)
            self.num_node+=1
            self.print_tree(node.right, feature_names, class_names, node.my_node_num)

    def export_graphviz(self, node, feature_names, class_names):
        self.num_node = 0
        self.dot_data = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica] ;\nedge [fontname=helvetica] ;\n"
        self.print_tree(node, feature_names, class_names, 0)
        self.dot_data += "}"
        return self.dot_data
        
if __name__ == "__main__":
    main()