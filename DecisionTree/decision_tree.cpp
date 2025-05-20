#include <bits/stdc++.h>
using namespace std;

// 節點結構
struct Node {
    bool isLeaf;            // 是否為葉節點
    int label;              // 如果是葉節點，儲存預測的類別標籤
    int featureIndex;       // 分裂所使用的特徵索引（對應第幾維特徵）
    double threshold;       // 分裂所使用的閾值
    Node* left;             // 左子節點指標 (特徵值 <= threshold)
    Node* right;            // 右子節點指標 (特徵值 > threshold)
    Node(): isLeaf(false), label(-1), featureIndex(-1), threshold(0.0), left(nullptr), right(nullptr) {}
};

// 全域變數
static vector<vector<int>> trainFeatures;  // 訓練特徵資料 [樣本索引][特徵索引]
static vector<int> trainLabels;            // 訓練標籤資料
static vector<int> testLabels;            // 測試標籤資料
static vector<int> trainPred;            // 測試標籤資料
static vector<int> testPred;            // 測試標籤資料
static int numFeatures = 0;               // 特徵維度 (預期 784)
static int numClasses = 0;                // 類別數量 (MNIST 預期 10)

// 建構決策樹的遞迴函式，參數為當前節點包含的資料索引集合
Node* buildTree(const vector<int>& dataIndexList) {
    // 節點初始化
    Node* node = new Node();
    // 如果當前節點的資料列表為空，返回空（不應發生此情況，僅防禦性處理）
    if (dataIndexList.empty()) {
        node->isLeaf = true;
        node->label = 0;
        return node;
    }
    // 檢查此節點資料是否全屬於同一類別
    int firstLabel = trainLabels[dataIndexList[0]];
    bool allSame = true;
    for (int idx : dataIndexList) {
        if (trainLabels[idx] != firstLabel) {
            allSame = false;
            break;
        }
    }
    if (allSame) {
        // 如果所有樣本標籤相同，直接作為葉節點標記該類別
        node->isLeaf = true;
        node->label = firstLabel;
        return node;
    }
    // 計算當前節點的類別分佈，用於計算不純度
    vector<int> labelCount(numClasses, 0); //記錄此節點每個類別出現的次數
    for (int idx : dataIndexList) {
        labelCount[trainLabels[idx]]++;
    }
    // 計算基尼不純度 (Gini impurity) = 1 - Σ((count[c]/N)^2)
    int N = dataIndexList.size();
    double parentImpurity = 1.0;
    for (int c = 0; c < numClasses; ++c) {
        if (labelCount[c] > 0) {
            double p = (double)labelCount[c] / N;
            parentImpurity -= p * p;
        }
    }
    // 若當前節點已無不純度（純淨單一類別），則成為葉節點（理論上已在 allSame 處理，此處再次檢查）
    if (parentImpurity == 0.0) {
        node->isLeaf = true;
        // 選擇具有最多樣本數的類別作為葉節點預測（其實 allSame 時已返回，這裡作保險）
        int majorityClass = distance(labelCount.begin(), max_element(labelCount.begin(), labelCount.end()));
        node->label = majorityClass;
        return node;
    }

    // 初始化最佳分裂變數
    double bestImpurityGain = 0.0;
    int bestFeatureIndex = -1;
    double bestThreshold = 0.0;

    // 嘗試每一個特徵作為分裂依據
    for (int f = 0; f < numFeatures; ++f) {
        // 提取該特徵在此節點資料中的值和標籤
        vector<pair<int,int>> featVals;
        featVals.reserve(N);
        for (int idx : dataIndexList) {
            featVals.emplace_back(trainFeatures[idx][f], trainLabels[idx]);
        }
        // 根據特徵值排序
        sort(featVals.begin(), featVals.end(),
             [](const pair<int,int>& a, const pair<int,int>& b){ return a.first < b.first; });
        // 若該特徵對所有樣本值都相同，則無法通過此特徵分裂，跳過
        if (featVals.front().first == featVals.back().first) {
            continue;
        }
        // 左右子集類別計數，用於計算不純度
        vector<int> leftCount(numClasses, 0);
        vector<int> rightCount = labelCount;  // 初始右側包含全部樣本
        int leftSize = 0;
        int rightSize = N;
        // 掃描可能的分裂點
        for (int i = 0; i < N - 1; ++i) {
            int val = featVals[i].first;
            int label = featVals[i].second;
            // 將當前樣本從右側移動到左側
            leftCount[label] += 1;
            rightCount[label] -= 1;
            leftSize++;
            rightSize--;
            // 檢查當前值與下一個值是否不同，只有在值改變的邊界才是候選分裂點
            if (val != featVals[i+1].first) {
                // 計算此分裂點的基尼不純度
                double giniLeft = 1.0;
                double giniRight = 1.0;
                for (int c = 0; c < numClasses; ++c) {
                    if (leftCount[c] > 0) {
                        double pL = (double)leftCount[c] / leftSize;
                        giniLeft -= pL * pL;
                    }
                    if (rightCount[c] > 0) {
                        double pR = (double)rightCount[c] / rightSize;
                        giniRight -= pR * pR;
                    }
                }
                double weightedGini = (double)leftSize / N * giniLeft + (double)rightSize / N * giniRight;
                double impurityGain = parentImpurity - weightedGini;
                // 如果此分裂帶來更大的不純度降低，則更新最佳分裂
                if (impurityGain > bestImpurityGain) {
                    bestImpurityGain = impurityGain;
                    bestFeatureIndex = f;
                    // 閾值取兩個不同值的中點
                    bestThreshold = ((double)val + (double)featVals[i+1].first) / 2.0;
                }
            }
        }
    }

    // 如果未找到有效的分裂（bestFeatureIndex仍為-1或增益為0），將此節點作為葉節點
    if (bestFeatureIndex == -1 || bestImpurityGain <= 1e-12) {
        node->isLeaf = true;
        // 選擇最多數據的類別作為葉節點類別
        int majorityClass = distance(labelCount.begin(), max_element(labelCount.begin(), labelCount.end()));
        node->label = majorityClass;
        return node;
    }

    // 使用找到的最佳特徵和閾值進行分裂
    node->featureIndex = bestFeatureIndex;
    node->threshold = bestThreshold;
    node->isLeaf = false;
    // 分別準備左右子節點的資料索引列表
    vector<int> leftIndices;
    vector<int> rightIndices;
    leftIndices.reserve(N);
    rightIndices.reserve(N);
    // 如果樣本在第 bestFeatureIndex 維度上的值 ≤ bestThreshold，就歸到左子樹；否則就到右子樹
    for (int idx : dataIndexList) {
        if ((double)trainFeatures[idx][bestFeatureIndex] <= bestThreshold) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    // 遞迴建立左子樹和右子樹
    node->left = buildTree(leftIndices);
    node->right = buildTree(rightIndices);
    return node;
}

// 使用訓練好的決策樹對單一樣本進行預測
int predict(const Node* node, const vector<int>& features) {
    const Node* cur = node;
    while (!cur->isLeaf) {
        // 根據當前節點的分裂規則，決定走向左或右子樹
        if ((double)features[cur->featureIndex] <= cur->threshold) {
            cur = cur->left;
        } else {
            cur = cur->right;
        }
        if (cur == nullptr) {
            // 安全檢查：不應該發生，如果發生則退出
            break;
        }
    }
    // 返回葉節點的預測類別
    return (cur ? cur->label : 0);
}
// 遞迴釋放整棵決策樹
void deleteTree(Node* node) {
    if (node == nullptr) return;
    // 先刪除左子樹
    deleteTree(node->left);
    // 再刪除右子樹
    deleteTree(node->right);
    // 最後刪除自己
    delete node;
}
int countNodes(Node* node) {
    if (!node) return 0;
    return 1 + countNodes(node->left) + countNodes(node->right);
}
void sumLeafDepth(Node* node, int depth, int& totalDepth, int& leafCount) {
    if (!node) return;
    if (node->isLeaf) {
        totalDepth += depth;
        leafCount++;
        return;
    }
    sumLeafDepth(node->left, depth + 1, totalDepth, leafCount);
    sumLeafDepth(node->right, depth + 1, totalDepth, leafCount);
}
double compute_macro_f1(const vector<int>& true_labels, const vector<int>& pred_labels) {
    int m = 10;
    double macro_f1 = 0.0;

    for (int c = 0; c < m; ++c) {
        int TP = 0, FP = 0, FN = 0;
        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (pred_labels[i] == c && true_labels[i] == c) TP++;
            else if (pred_labels[i] == c && true_labels[i] != c) FP++;
            else if (pred_labels[i] != c && true_labels[i] == c) FN++;
        }

        double precision = (TP + FP == 0) ? 0 : (double)TP / (TP + FP);
        double recall = (TP + FN == 0) ? 0 : (double)TP / (TP + FN);
        double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

        macro_f1 += f1;
    }

    return macro_f1 / m;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    
    // 檔案名稱，可根據需要修改或使用命令列參數
    string trainFile = "mnist_train.csv";
    string testFile = "mnist_test.csv";
    ifstream finTrain(trainFile);
    if (!finTrain) {
        cerr << "Cannot open the file: " << trainFile << "\\n";
        return 1;
    }
    ifstream finTest(testFile);
    if (!finTest) {
        cerr << "Cannot open the file: " << testFile << "\\n";
        return 1;
    }

    // 讀取訓練資料
    string line;
    while (getline(finTrain, line)) {
        if (line.size() == 0) continue;  // 跳過空行
        string token;
        stringstream ss(line);
        vector<int> features;
        features.reserve(784);
        // 分割逗號，讀取所有欄位
        while (getline(ss, token, ',')) {
            if (token.size() == 0) {
                // 若有空欄位，視為0
                features.push_back(0);
            } else {
                // 將字串轉成整數
                features.push_back(stoi(token));
            }
        }
        // 最後一個值為標籤
        int label = features.back();
        features.pop_back();
        // 如果特徵數不足784，補齊0
        if (features.size() < 784) {
            features.resize(784, 0);
        }
        // 記錄此樣本的特徵和標籤
        trainFeatures.push_back(features);
        trainLabels.push_back(label);
    }
    finTrain.close();
    // 特徵維度設定為784（或根據第一筆資料長度）
    if (!trainFeatures.empty()) {
        numFeatures = trainFeatures[0].size();
    } else {
        numFeatures = 784;
    }
    // 推斷類別數量（例如找出最大標籤值）
    int maxLabel = -1;
    for (int lab : trainLabels) {
        if (lab > maxLabel) maxLabel = lab;
    }
    numClasses = maxLabel + 1;
    if (numClasses < 2) numClasses = 2;  // 至少設定為2類，以防只有單一類別的極端情況

    // 讀取測試資料
    vector<vector<int>> testFeatures;
    while (getline(finTest, line)) {
        if (line.size() == 0) continue;
        string token;
        stringstream ss(line);
        vector<int> features;
        features.reserve(784);
        while (getline(ss, token, ',')) {
            if (token.size() == 0) {
                features.push_back(0);
            } else {
                features.push_back(stoi(token));
            }
        }
        // 最後一個值為標籤
        int label = features.back();
        features.pop_back();
        // 如果特徵數不足784，補齊0
        if (features.size() < 784) {
            features.resize(784, 0);
        }
        // 記錄此樣本的特徵和標籤
        testFeatures.push_back(features);
        testLabels.push_back(label);
    }
    finTest.close();
    // 如未能從訓練資料獲得特徵數，從測試資料設定
    if (numFeatures == 0 && !testFeatures.empty()) {
        numFeatures = testFeatures[0].size();
    }

    using namespace chrono;
    auto start = high_resolution_clock::now();  // 開始計時
    // 建構決策樹模型
    vector<int> allIndices;
    allIndices.reserve(trainFeatures.size());
    for (int i = 0; i < (int)trainFeatures.size(); ++i) {
        allIndices.push_back(i);
    }
    Node* root = buildTree(allIndices);

    auto end = high_resolution_clock::now();    // 結束計時
	duration<double> duration = end - start;

    // 對訓練集進行預測並輸出結果
    ofstream foutTrain("result_train.csv");
    for (size_t i = 0; i < trainFeatures.size(); ++i) {
        int predLabel = predict(root, trainFeatures[i]);
        trainPred.push_back(predLabel);
        foutTrain << predLabel << "\n";
    }
    foutTrain.close();

    // 對測試集進行預測並輸出結果
    ofstream foutTest("result_test.csv");
    for (size_t i = 0; i < testFeatures.size(); ++i) {
        int predLabel = predict(root, testFeatures[i]);
        testPred.push_back(predLabel);
        foutTest << predLabel << "\n";
    }
    foutTest.close();

    // 計算 Macro F1-score
    double f1_train = compute_macro_f1(trainLabels, trainPred);
    double f1_test = compute_macro_f1(testLabels, testPred);
    cout << "Train Macro F1 Score: " << f1_train << endl;
    cout << "Test  Macro F1 Score: " << f1_test << endl;

    //計算節點數量
    cout << "Total nodes in tree: " << countNodes(root) << endl;
    //計算平均深度
    int totalDepth = 0, leafCount = 0;
    sumLeafDepth(root, 0, totalDepth, leafCount);
    double avgLeafDepth = (double)totalDepth / leafCount;
    cout << "Average leaf depth: " << avgLeafDepth << endl;
    cout << "Node size:" << sizeof(Node) << endl;
    
    deleteTree(root);// 釋放決策樹節點佔用的記憶體
    
    cout << "running time: " << duration.count() << endl;
    

    return 0;
}
