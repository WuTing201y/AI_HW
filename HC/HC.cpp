#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <ctime>
#include <string>
#include <chrono>
using namespace std;

struct City{
	double x;
	double y;
};

double calculate_distance(const City& a, const City& b)
{
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	return sqrt(dx*dx + dy*dy);
}

double calculate_total_dis(const vector<int>& path, const vector<vector<double>>& dist_map)
{
	double total = 0;
	int n = path.size(); //要記得定義n
	for(int i = 0; i < n - 1; i++){
		total += dist_map[path[i]][path[i+1]];		
	}
	total += dist_map[path[n-1]][path[0]];
	return total;
}

double hc(vector<int>& path, const vector<vector<double>>& dist_map, int n, int D)
{
	for(int i = 0; i < n; i++){
		path.push_back(i);
	}
	//打亂順序
	mt19937 g(time(0));
	shuffle(path.begin(), path.end(), g);

	//目前最佳路徑path，與其距離cur_dis
	double cur_dis = calculate_total_dis(path, dist_map);

	//建立亂數分布器
	uniform_int_distribution<> dist(0, n-1);

	for (int t = 0; t < 1000*D; t++) {
	    int i = dist(g);
	    int j = dist(g);
	    while (i == j)
	        j = dist(g);

	    swap(path[i], path[j]);
	    double new_dis = calculate_total_dis(path, dist_map);

	    if (new_dis < cur_dis){
	        cur_dis = new_dis;  // 接受
	    	//cout << "accept the " << t+1 << " times change, new distance = " << cur_dis << endl;
	    }
	    else{
	        swap(path[i], path[j]);  // 還原
	    }
	}
	
	return cur_dis;
}

int main()
{
	using namespace chrono;

	vector<int> dim = {50, 100, 200, 500, 1000};

	//讀取資料
	for(int D : dim){
		vector<City> cities;

		string filename = "TSP_Dim=" + to_string(D) + ".txt";
		ifstream fin(filename);
		if(!fin){
			cerr << "Cannot open the file." << endl;
			continue;
		}

		int id;
		double x, y;
		while(fin >> id >> x >> y){
			City c;
			c.x = x;
			c.y = y;
			cities.push_back(c);
		}

		//建立一個 n × n 的二維陣列，每一格初始值都是 0.0，儲存每兩個城市間的距離
		int n = cities.size();
		vector<vector<double>> dist_map(n, vector<double>(n, 0.0));
		for(int i = 0; i < n-1; i++){
			for(int j = i+1; j < n; j++){
				dist_map[i][j] = dist_map[j][i] = calculate_distance(cities[i], cities[j]);
			}
		}

		//建立初始順序
		vector<int> path;
		auto start = high_resolution_clock::now();  // 開始計時
		double best_dis = hc(path, dist_map, n, D);
		auto end = high_resolution_clock::now();    // 結束計時
		duration<double> duration = end - start;
		
		 //儲存距離
		string dist_out = "result_distance_D=" + to_string(D) + ".txt";
		ofstream fout_dist(dist_out, ios::app);
		fout_dist << best_dis << endl;
		fout_dist.close();

		// 儲存時間
		string time_out = "result_time_D=" + to_string(D) + ".txt";
		ofstream fout_time(time_out, ios::app);
		fout_time << duration.count() << " seconds" << endl;
		fout_time.close();

		string out_filename = "output_Dim=" + to_string(D) + ".txt";
		ofstream fout(out_filename);

		for(int city : path){
			fout << city + 1 << " "; //對齊城市編號
		}
		fout << endl;
		fout.close();
		
		//cout << "completed" << endl;
	}
}
