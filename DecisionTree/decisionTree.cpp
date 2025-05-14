#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace std;
using Vec  = std::vector<int>;         // 一筆 (離散) 特徵(每row由左至右)
using Mat  = std::vector<Vec>;         // 多筆特徵 特徵矩陣
using VecL = std::vector<int>;         // 所有樣本的標籤向量 最後一col的標籤

struct Node{
    bool is_leaf = false;
    int label = -1;
    int split_attr = -1;
    unordered_map<int, unique_ptr<Node>> child;
};
class DecisionTree{
public:
    void fit(const Mat& X, const VecL& y, const Vec& attrs)
    {
        Vec all_idx(X.size());
        iota(all_idx.begin(), all_idx.end(), 0); //設定好每筆樣本的索引
        root = decisionTree(all_idx, attrs, X, y);
    }
    int predict(const Vec& x) const
    {
        return predictRec(root.get(), x);
    }
private:
    unique_ptr<Node> root;
    unique_ptr<Node> decisionTree(const Vec& ex_idx, const Vec& attrs, const Mat& X, const VecL& y)
    {
        // if same class     → return class
        if(allSameClass(ex_idx, y)){
            auto leaf = make_unique<Node>();
            leaf->is_leaf = true;
            leaf->label = y[ex_idx[0]]; //因為標籤都一樣，拿哪一筆都行，而拿第一筆最省事
            return leaf;
        }
        // if attributes empty → plurality(examples)
        if(attrs.empty()){
            auto leaf = make_unique<Node>();
            leaf->is_leaf = true;
            leaf->label = pluralityValue(ex_idx, y);
            return leaf;
        }
        // else
        int best_attr = -1;
        double best_gain = -1e9;
        for(int a : attrs){
            double g = infoGain(a, ex_idx, X, y);
            if(g > best_gain){
                best_gain = g;
                best_attr = a;
            }
        }
        auto node = make_unique<Node>();
        node->split_attr = best_attr;
        node->label = pluralityValue(ex_idx, y);

        //對attr分割
        unordered_map<int , Vec> subsets;
        for(int idx : ex_idx){
            subsets[X[idx][best_attr]].push_back(idx);
        }
        Vec new_attrs;
        for(int a : attrs){
            if(a!=best_attr){
                new_attrs.push_back(a);
            }
        }
        for(auto& kv : subsets){
            int v = kv.first;  //key
            const Vec& sub_idx = kv.second; //value
            node->child[v] = decisionTree(sub_idx, new_attrs, X, y);
        }
        return node;
    }
    int pluralityValue(const Vec& ex_idx, const VecL& y) const
    {
        if(ex_idx.empty()) return 0;
        unordered_map<int, int> cnt;
        for(int i : ex_idx) cnt[y[i]]++;
        return max_element(cnt.begin(), cnt.end(),
                [](auto& a, auto& b){return a.second < b.second;}
                )->first;           
    }
    bool allSameClass(const Vec& ex_idx, const VecL& y) const
    {
        int first = y[ex_idx[0]];
        for(int i : ex_idx){
            if(y[i] != first) return false;
        }
        return true;
    }
    double entropy(const Vec& ex_idx, const VecL& y) const
    {
        unordered_map<int, int> cnt;
        for(int i : ex_idx) cnt[y[i]]++;
        double H = 0.0, N = ex_idx.size();
        for(const auto& kv: cnt){
            int c = kv.second;  //次數
            double p = static_cast<double>(c) / N;   //機率
            H -= p*log2(p);
        }
        return H;
    }
    double infoGain(int attr, const Vec& ex_idx, const Mat& X, const VecL& y) const
    {
        double H_before = entropy(ex_idx, y);
        unordered_map<int, Vec> subsets;
        for(int idx : ex_idx){
            subsets[X[idx][attr]].push_back(idx); 
        }
        double H_after = 0.0, N = static_cast<double>(ex_idx.size());
        for(const auto& it : subsets){
            const Vec& sub = it.second;
            double w = static_cast<double>(sub.size()) / N;
            H_after += w * entropy(sub, y);
        }
        return H_before - H_after;
    }
    int predictRec(const Node* node, const Vec& x) const
    {
        if(node->is_leaf) return node->label;
        int v = x[node->split_attr];
        auto it = node->child.find(v);
        if(it == node->child.end()) return node->label;
        return predictRec(it->second.get(), x);
    }
};

// X代表所有特徵矩陣，Y代表每筆資料的類別標籤
void readCSV(const string& path, Mat& X, VecL& Y)
{
    ifstream fin(path);
    if(!fin){
        cerr << "Cannot open the file." << endl;
    }
    string line;
    while(getline(fin, line)){
        stringstream ss(line);
        string cell;
        Vec row;

        while(getline(ss, cell, ',')){
            row.push_back(stoi(cell));
        }
        Y.push_back(row.back()); //取得最後一個label
        row.pop_back(); //去掉最後一個label
        X.push_back(move(row));  //move()可將row的東西搬到X裡，而row被清空
    }
}
void savePredict(const string& out, const Vec& pred)
{
    ofstream fout(out);
    for(int p : pred) fout << p << endl;
}
int main()
{
    Mat X_train, X_test;
    VecL Y_train, Y_trash;
    readCSV("mnist_train.csv", X_train, Y_train);
    readCSV("mnist_test.csv", X_test, Y_trash);

    Vec attrs(X_train[0].size());
    iota(attrs.begin(), attrs.end(), 0);

    DecisionTree dt;
    dt.fit(X_train, Y_train, attrs);

    Vec pred_train;
    for(auto& x : X_train){
        pred_train.push_back(dt.predict(x));
    }
    savePredict("result_train.csv", pred_train);

    Vec pred_test;
    for(auto& x : X_test){
        pred_test.push_back(dt.predict(x));
    }
    savePredict("result_test.csv", pred_test);
    cout << "done" << endl;
}