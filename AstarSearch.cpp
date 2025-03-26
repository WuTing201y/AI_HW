#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>

using namespace std;
vector<vector<int>> Clause;

//readCSV
vector<vector<int>> readCSV(const string &filename, int &D){
    Clause.clear(); //初始化
    ifstream file(filename);

    if(!file.is_open()){
        cerr << "Cannot open the file: " << filename << endl; //cerr用於錯誤輸出
        return vector<vector<int>>();
    }

    string line;
    while(getline(file, line)){
        stringstream ss(line);
        vector<int> row;
        string token;

        while(getline(ss, token, ',')){
            if(!token.empty() && token.back() == '\r'){
                token.pop_back(); //刪掉容器中最後一個元素
            }
            try{
                row.push_back(stoi(token));
            }
            catch(const invalid_argument &e){
                cerr << "Invalid integer in file: " << token << endl;
                continue; //忽略錯誤資料
            }
        }
        if(!row.empty()){
            Clause.push_back(row);//把每一行row放到vector的data裡面(同時data裡面包含3個
        }

    }
    file.close();
    D = 50;
    return Clause;
}

struct Node{ //節點
    vector<int> assignment; //該變量是否已經賦值了(有沒有確認sign是1還是0了?)
    int g; // real cost
    int h; // 估計cost
    int f; //total cost

    Node(int D) : assignment(D, -1), g(0), h(0), f(0){}

    bool operator < (const Node& other) const{
        return f > other.f;  //以最小的f為優先
    }
};

int heuristic(const Node &node, const vector<vector<int>> &clauses)
{   //估算從當前狀態到滿足所有子句的目標狀態的「成本」

    int unsatisfied = 0;
    for(const auto& clause : clauses){
        bool clause_ok = false;
        for(int i = 0; i < 3; i++){
            int element = clause[i];  //Ex: clause=[-1, 3, -5]
            int sign = (element > 0) ? 1 : -1; //sign 分別為 -1, 1, -1
            int var_idx = abs(element) - 1; //轉成0-base索引值，idx 分別為 0, 2, 4
            int var_value = node.assignment[var_idx]; //assignment初始都是-1,

            if(var_value == -1) continue; //尚未賦值


            int literal = (sign == 1) ? var_value : (1 - var_value);
            if(literal == 1){
                clause_ok = true;
                break;
            }
        }
        if(!clause_ok) unsatisfied++;
    }
    return unsatisfied;
}
int astar(int D)
{
    priority_queue<Node> pq;
    Node start(D);
    start.h = heuristic(start, Clause);
    start.f = start.g + start.h;
    pq.push(start);

    while(!pq.empty()){
        Node current = pq.top();
        pq.pop();

        /*//測試輸出
        cout << "Processing node: ";
        for (int v : current.assignment) cout << v << " ";
        cout << "g=" << current.g << " h=" << current.h << endl;
        */

        //檢查是否所有變量已賦值
        if(current.g == D){
            // 驗證是否滿足所有子句
            bool all_ok = true;
            for(const auto& clause : Clause){
                bool clause_ok = false;

                for(int i = 0; i < 3; i++){
                    int element = clause[i];  //Ex: clause=[-1, 3, -5]
                    int sign = (element > 0) ? 1 : -1; //sign 分別為 -1, 1, -1
                    int var_idx = abs(element) - 1; //轉成0-base索引值，idx 分別為 0, 2, 4
                    int var_value = current.assignment[var_idx]; //assignment初始都是-1,

                    if(var_value == -1) continue; //尚未賦值


                    int literal = (sign == 1) ? var_value : (1 - var_value);
                    if(literal == 1){
                        clause_ok = true;
                        break;
                    }
                }
                if(!clause_ok){
                    all_ok = false;
                    break;
                }
            }
            if(all_ok){
                for(int val : current.assignment){
                    cout << val << " ";
                }
                ofstream out("C:/Astar/result.txt");
                for(int val : current.assignment){
                    out << val << " ";
                }
                out.close();
                return 1; //有解
            }
            continue;
        }

        //擴展子節點:為下一個未賦值的變量嘗試0和1
        int next_var = current.g;
        for (int val : {0, 1}) {
            Node child = current;
            child.assignment[next_var] = val;
            child.g = current.g + 1;

            //剪枝:檢查是否有clause的所有變量已賦值且不滿足
            bool valid = true;
            for(const auto &clause : Clause){
                bool clause_ok = false;
                bool all_assigned = true;
                for(int i = 0; i < 3; i++){
                    int element = clause[i];
                    int var_idx = abs(element) - 1;
                    int var_value = child.assignment[var_idx];

                    if(var_value == -1){
                        all_assigned = false;
                        break;
                    }
                }
                if(!all_assigned) continue;

                //// 若所有變量已賦值但子句不滿足
                for(int k = 0; k < 3; k++){
                    int element = clause[k];
                    int sign = (element > 0) ? 1 : -1;
                    int var_idx = abs(element) - 1;
                    int var_value = child.assignment[var_idx];
                    int literal = (sign == 1) ? var_value : (1 - var_value);

                    if(literal == 1){
                        clause_ok = true;
                        break;
                    }
                }
                if(!clause_ok){
                    valid = false;
                    break;
                }
            }
            if(valid){
                child.h = heuristic(child, Clause);
                child.f = child.g + child.h;
                pq.push(child);
            }
        }
    }
    ofstream out("C:/Astar/TEST_DATA/result.txt");
    out << "No solution";
    out.close();
    return 0;
}

int main()
{
    int D;
    Clause = readCSV("3SAT_Dim=50.csv", D);
    /*確認是否有讀入
    for(const auto &row : csv_data){
        for(const auto &cell : row){
            cout << cell << " ";
        }
        cout << endl;
    }
    */
    int result = astar(D);
    cout << (result ? "Solution found!" : "No solution") << endl;
    return 0;
}
