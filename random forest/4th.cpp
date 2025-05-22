/****************  random_forest.cpp  ****************/
#include <bits/stdc++.h>
using namespace std;

/* ==== 全域常數與別名 ==== */
constexpr int NUM_CLASSES = 10;       // MNIST 0-9
constexpr int FEATURE_DIM = 784;      // 28×28
using VecI  = vector<int>;
using VecD  = vector<double>;
using MatI  = vector<VecI>;

/* ==== 資料結構 ==== */
struct Node {
    bool leaf   = false;
    int  label  = -1;

    /* 內部節點 */
    int    feat = -1;
    double thr  = 0.0;
    Node  *left = nullptr, *right = nullptr;
};
/* 取得單棵樹的節點數 */
static int countNodes(const Node* n) {
    if (!n) return 0;
    return 1 + countNodes(n->left) + countNodes(n->right);
}
static void sumLeafDepth(const Node* n, int depth,
                  long long& totalDepth, long long& leafCnt)
{
    if (!n) return;
    if (n->leaf) {           // 到葉子：累積深度與計數
        totalDepth += depth;
        ++leafCnt;
        return;
    }
    sumLeafDepth(n->left , depth + 1, totalDepth, leafCnt);
    sumLeafDepth(n->right, depth + 1, totalDepth, leafCnt);
}

static void freeNode(Node* p){
    if(!p) return;
    freeNode(p->left);
    freeNode(p->right);
    delete p;
}

/* ==== 工具函式 ==== */
double gini(const array<int,NUM_CLASSES>& cnt, int n) {
    if (n==0) return 0.0;
    double g = 1.0;
    for (int c=0;c<NUM_CLASSES;++c) {
        if (cnt[c]==0) continue;
        double p = (double)cnt[c]/n;
        g -= p*p;
    }
    return g;
}

/* ==== 決策樹 ==== */
class DecisionTree {
public:
    DecisionTree(int maxDepth,int minLeaf,int maxFeat,
                 mt19937& rng,const MatI& X,const VecI& y)
        : depthLimit(maxDepth), minLeafSize(minLeaf), maxFeatures(maxFeat),
          rnd(rng), Xref(X), yref(y) {}

    Node* build(const VecI& idx,int depth=0) {
        Node* node = new Node();
        int n = idx.size();
        /* 類別計數 */
        array<int,NUM_CLASSES> cnt{}; cnt.fill(0);
        for(int id:idx) cnt[yref[id]]++;
        /* 若樣本屬同類或達深度/葉大小門檻 → 葉節點 */
        int majority = distance(cnt.begin(),
                         max_element(cnt.begin(),cnt.end()));
        if (depth>=depthLimit || n<=minLeafSize || cnt[majority]==n){
            node->leaf=true; node->label=majority; return node;
        }
        double parentGini = gini(cnt,n);

        /* 隨機抽 maxFeatures 個特徵 */
        vector<int> featPool(FEATURE_DIM);
        iota(featPool.begin(),featPool.end(),0);
        shuffle(featPool.begin(),featPool.end(),rnd);
        featPool.resize(maxFeatures);

        /* 搜最佳分裂 */
        double bestGain=0; int bestF=-1; double bestThr=0;
        VecI bestLeft,bestRight;

        for(int f:featPool){
            /* 蒐集(值,idx)並排序 */
            vector<pair<int,int>> vals;
            vals.reserve(n);
            for(int id:idx) vals.emplace_back(Xref[id][f], id);
            sort(vals.begin(),vals.end());

            array<int,NUM_CLASSES> leftCnt{}; leftCnt.fill(0);
            array<int,NUM_CLASSES> rightCnt = cnt;
            int nl=0,nr=n;

            for(int i=0;i<n-1;++i){
                int id = vals[i].second;
                int lab= yref[id];
                leftCnt[lab]++;  rightCnt[lab]--;
                nl++; nr--;
                if(vals[i].first==vals[i+1].first) continue; // 值相同不切

                double gl = gini(leftCnt,nl);
                double gr = gini(rightCnt,nr);
                double gain = parentGini - ((double)nl/n)*gl - ((double)nr/n)*gr;
                if(gain>bestGain){
                    bestGain=gain; bestF=f;
                    bestThr=(vals[i].first+vals[i+1].first)/2.0;
                }
            }
        }
        /* 若無有效分裂→葉節點 */
        if(bestF==-1 || bestGain<=1e-7){
            node->leaf=true; node->label=majority; return node;
        }

        /* 生成左右索引 */
        VecI leftIdx,rightIdx;
        leftIdx.reserve(idx.size());
        rightIdx.reserve(idx.size());
        for(int id:idx){
            if(Xref[id][bestF]<=bestThr) leftIdx.push_back(id);
            else                         rightIdx.push_back(id);
        }

        node->feat=bestF; node->thr=bestThr;
        node->left = build(leftIdx, depth+1);
        node->right= build(rightIdx,depth+1);
        return node;
    }
    int predict(Node* node,const VecI& x) const{
        const Node* cur=node;
        while(!cur->leaf){
            cur = (x[cur->feat]<=cur->thr)?cur->left:cur->right;
        }
        return cur->label;
    }
private:
    int  depthLimit, minLeafSize, maxFeatures;
    mt19937& rnd;
    const MatI& Xref; const VecI& yref;
};

/* ==== Random Forest ==== */
class RandomForest {
public:
    RandomForest(int nTrees,int maxDepth,int minLeaf,int maxFeat)
        : T(nTrees), depth(maxDepth), minLeaf(minLeaf), maxFeat(maxFeat) {}

    void fit(const MatI& X,const VecI& y){
        rng.seed(42);
        uniform_int_distribution<int> uni(0,X.size()-1);
        for(int t=0;t<T;++t){
            /* bootstrap 樣本索引 */
            VecI sampleIdx;
            sampleIdx.reserve(X.size());
            for(size_t i=0;i<X.size();++i) sampleIdx.push_back(uni(rng));

            DecisionTree tree(depth,minLeaf,maxFeat,rng,X,y);
            roots.push_back(tree.build(sampleIdx));
            trees.push_back(std::move(tree));
        }
    }
    size_t nodeCount() const {
        size_t cnt = 0;
        for (auto r: roots) cnt += countNodes(r);
        return cnt;
    }
    double averageLeafDepth() const {
        long long td=0, lc=0;
        for (auto r: roots) sumLeafDepth(r,0,td,lc);
        return lc ? (double)td/lc : 0.0;
    }
    int predict(const VecI& x) const{
        array<int,NUM_CLASSES> vote{}; vote.fill(0);
        for(int i=0;i<T;++i){
            int lab = trees[i].predict(roots[i],x);
            vote[lab]++;
        }
        return distance(vote.begin(),max_element(vote.begin(),vote.end()));
    }
    
private:
    int T,depth,minLeaf,maxFeat;
    mutable vector<Node*> roots;
    vector<DecisionTree> trees;
    mt19937 rng;
};

/* ==== CSV 讀入 ==== */
bool loadCSV(const string& file, MatI& X, VecI& y){
    ifstream fin(file);
    if(!fin){
        cerr << "Cannot open the file: " << file << "\\n";
        return false;
    }
    string line; getline(fin,line); // skip header
    while(getline(fin,line)){
        stringstream ss(line); string tok; VecI vec;
        vec.reserve(FEATURE_DIM);
        while(getline(ss,tok,',')) vec.push_back(stoi(tok));
        y.push_back(vec.back()); vec.pop_back();
        if(vec.size()<FEATURE_DIM) vec.resize(FEATURE_DIM,0);
        X.push_back(std::move(vec));
    }
    return true;
}

/* ==== Macro-F1 計算 ==== */
double macroF1(const VecI& yt,const VecI& yp){
    int tp[NUM_CLASSES]={},fp[NUM_CLASSES]={},fn[NUM_CLASSES]={};
    for(size_t i=0;i<yt.size();++i){
        int t=yt[i], p=yp[i];
        if(t==p) tp[t]++; else {fp[p]++; fn[t]++;}
    }
    double f1=0;
    for(int c=0;c<NUM_CLASSES;++c){
        double prec = (tp[c]+fp[c])? (double)tp[c]/(tp[c]+fp[c]) : 0;
        double rec  = (tp[c]+fn[c])? (double)tp[c]/(tp[c]+fn[c]) : 0;
        double f = (prec+rec)? 2*prec*rec/(prec+rec) : 0;
        f1+=f;
    }
    return f1/NUM_CLASSES;
}



/* ==== 主程式 ==== */
int main(){
    string trainFile = "mnist_train.csv";
    string testFile = "mnist_test.csv";
    MatI Xtrain,Xtest; VecI ytrain,ytest;
    loadCSV(trainFile,Xtrain,ytrain);
    loadCSV(testFile,Xtest ,ytest );

    using namespace chrono;
    auto start = high_resolution_clock::now();  // 開始計時

    int nTrees=50, maxDepth=12;
    int minLeaf=5, maxFeat=sqrt(FEATURE_DIM);

    RandomForest rf(nTrees,maxDepth,minLeaf,maxFeat);
    rf.fit(Xtrain,ytrain);

    VecI predTrain, predTest;
    for(const auto& x:Xtrain) predTrain.push_back(rf.predict(x));
    for(const auto& x:Xtest ) predTest .push_back(rf.predict(x));

    auto end = high_resolution_clock::now();    // 結束計時
	duration<double> duration = end - start;

    ofstream("result_train.csv")<<std::accumulate(predTrain.begin(),predTrain.end(),string(""),
        [](string a,int v){return a+to_string(v)+'\n';});
    ofstream("result_test.csv")<<std::accumulate(predTest.begin(),predTest.end(),string(""),
        [](string a,int v){return a+to_string(v)+'\n';});

    cout<<"Train F1="<<macroF1(ytrain,predTrain)<<"\n";
    cout<<"Test  F1="<<macroF1(ytest ,predTest )<<"\n";

    cout << "Total nodes in tree:  = " << rf.nodeCount() << '\n';
    cout << "Average leaf depth: " << rf.averageLeafDepth() << '\n';
    cout << "Node size:" << sizeof(Node) << endl;
    cout << "running time: " << duration.count() << endl;
    return 0;
}
