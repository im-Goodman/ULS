#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <regex>
#include <unordered_map>
#include <limits>
#include <chrono>
#include <queue>

using namespace std;
using namespace std::chrono;

struct ScheduleEvent {
    int task;
    double start;
    double end;
    int proc;
    string toString() const {
        return std::to_string(task) + " " + std::to_string(start) + " " + std::to_string(end) + " " + std::to_string(proc);
    }
    ScheduleEvent(int t, double s, double e, int p)
        : task(t), start(s), end(e), proc(p) {}
    ScheduleEvent()
        : task(-1), start(-1), end(-1), proc(-1) {}
};

struct NodeInfo {
    int jobId;
    int baseId;
    std::vector<int> predecessors;
    int id;
    NodeInfo()
        : jobId(-1), baseId(-1), predecessors({}), id(-1) {}
    NodeInfo(int job_id, int base_id, std::vector<int> preds, int node_id)
        : jobId(job_id), baseId(base_id), predecessors(preds), id(node_id) {}
};


class Graph {
public:
    vector<int> nodes;
    map<pair<int, int>, double> edgeWeights;
    map<pair<int, int>, double> avgWeights;
    map<int, set<int>> parentSets;
    map<int, set<int>> successors;
    map<int, map<string, double>> dict;

    Graph() : nodes(), edgeWeights(), parentSets(), successors(), dict() {}

    void readFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "无法打开文件。" << endl;
            return;
        }

        string line;
        string section;

        while (getline(file, line)) {
            if (line.empty()) continue;

            if (line.find("Nodes:")!= string::npos) {
                section = "Nodes";
                continue;
            } else if (line.find("Edge Weights:")!= string::npos) {
                section = "Edge Weights";
                continue;
            } else if (line.find("Avg Weights:")!= string::npos) {
                section = "Avg Weights";
                continue;
            } else if (line.find("Parent Sets:")!= string::npos) {
                section = "Parent Sets";
                continue;
            } else if (line.find("Successor Sets:")!= string::npos) {
                section = "Successor Sets";
                continue;
            }

            if (section == "Nodes") {
                regex pattern("(\\d+):\\{t:(.*?)\\} \\{t1:(.*?)\\} \\{sd:(.*?)\\}");
                smatch match;
                if (regex_search(line, match, pattern)) {
                    int node = stoi(match[1]);
                    nodes.push_back(node);
                    dict[node] = map<string, double>();
                    if (match[2] != "-1") {
                        double tValue = stod(match[2]);
                        dict[node]["t"] = tValue;
                    }
                    if (match[3] != "-1") {
                        double t1Value = stod(match[3]);
                        dict[node]["t1"] = t1Value;
                    }
                    if (match[4] != "-1") {
                        double sdValue = stod(match[4]);
                        dict[node]["sd"] = sdValue;
                    }
                }
            } else if (section == "Edge Weights") {
                istringstream iss(line);
                int from, to;
                char ch;
                double weight;
                char colon;
                iss >> from >> ch>> to >> colon >> weight;
                edgeWeights[{from, to}] = weight;
            } else if (section == "Avg Weights") {
                istringstream iss(line);
                int from, to;
                char ch;
                double weight;
                char colon;
                iss >> from >> ch>> to >> colon >> weight;
                avgWeights[{from, to}] = weight;
            } else if (section == "Parent Sets") {
                istringstream iss(line);
                int node;
                iss >> node;
                char ch;
                iss>>ch;
                string parentsStr;
                iss >> parentsStr;
                parentsStr.erase(0, parentsStr.find_first_not_of(" \t\n\r\f\v"));
                getline(iss, parentsStr);
                set<int> parents;
                if (parentsStr != "null") {
                    istringstream iss1(parentsStr);
                    string token;
                    while (getline(iss1, token, ',')) {
                        int value = stoi(token);
                        parents.insert(value);
                    }
                    if (parents.empty()) {
                        parents.insert(stoi(parentsStr));
                    }
                }
                parentSets[node] = parents;
            } else if (section == "Successor Sets") {
                istringstream iss(line);
                int node;
                iss >> node;
                char ch;
                iss>>ch;
                string parentsStr;
                iss >> parentsStr;
                parentsStr.erase(0, parentsStr.find_first_not_of(" \t\n\r\f\v"));
                getline(iss, parentsStr);
                set<int> parents;
                if (parentsStr != "null") {
                    istringstream iss1(parentsStr);
                    string token;
                    while (getline(iss1, token, ',')) {
                        int value = stoi(token);
                        parents.insert(value);
                    }
                    if (parents.empty()) {
                        parents.insert(stoi(parentsStr));
                    }
                }
                successors[node] = parents;
            }
        }

        file.close();
    }
};

vector<int> load1(string path) {
    vector<int> sorted_nodes;
    ifstream file(path);
    if (file.is_open()) {
        string line;
        getline(file, line);
        // 去除首尾的方括号
        line = line.substr(1, line.length() - 2);
        istringstream iss(line);
        string numberStr;
        while (getline(iss, numberStr, ',')) {
            sorted_nodes.push_back(stoi(numberStr));
        }
        file.close();
    } else {
        cerr << "无法打开文件 " << path << endl;
    }
    file.close();
    return sorted_nodes;
}

unordered_map<int, NodeInfo> getDict(string path){
    ifstream file1(path);

    string content((istreambuf_iterator<char>(file1)), istreambuf_iterator<char>());

    regex pattern("(\\d+):(\\d+) (\\d+) \\[(.*)\\] (\\d+)");
    sregex_iterator it(content.begin(), content.end(), pattern);
    sregex_iterator end;

    unordered_map<int, NodeInfo> dict1;

    while (it!= end) {
        int node = stoi(it->str(1));
        int jobId = stoi(it->str(2));
        int baseId = stoi(it->str(3));
        string predecessorsStr = it->str(4);
        int id = stoi(it->str(5));

        vector<int> predecessors;
        if (!predecessorsStr.empty()) {
            istringstream predecessorsIss(predecessorsStr);
            int predecessor;
            while (predecessorsIss >> predecessor) {
                if (predecessorsIss.peek() == ',') {
                    predecessorsIss.ignore();
                }
                predecessors.push_back(predecessor);
            }
        }

        dict1[node] = NodeInfo{jobId, baseId, predecessors, id};
        ++it;
    }
    file1.close();
    return dict1;
}

NodeInfo getTask(int node, unordered_map<int, NodeInfo>& dict1) {
    if (dict1.find(node) != dict1.end()) {
        return dict1[node];
    }
    vector<int> nu;
    return NodeInfo(-1, -1, nu, -1);
}

void output(string str, unordered_map<int, tuple<int, int, vector<int>, double, double, int>>& dict_output) {
    ofstream outputFile(str);
    for (const auto& pair : dict_output) {
        int key = pair.first;
        tuple<int, int, vector<int>, double, double, int> value = pair.second;
        outputFile << key << " ";
        outputFile << get<0>(value) << " ";
        outputFile << get<1>(value) << " ";
        if (std::get<2>(value).size() == 0) {
            outputFile << "null" << " ";
        } else {
            for (const auto& element : get<2>(value)) {
                outputFile << element << " ";
            }
        }
        outputFile << get<3>(value) << " ";
        outputFile << get<4>(value) << " ";
        outputFile << get<5>(value) << endl;
    }
    outputFile.close();
}

void output1(string str, unordered_map<int, vector<ScheduleEvent>>& procSchedules) {
    std::ofstream outputFile(str);

    for (const auto& pair : procSchedules) {
        for (const ScheduleEvent& event : pair.second) {
            outputFile << pair.first << " " << event.toString() << std::endl;
        }
    }
    outputFile.close();
}

void output2(string str, unordered_map<int, ScheduleEvent>& taskSchedules) {
    std::ofstream outputFile(str);

    for (const auto& outerPair : taskSchedules) {
        outputFile << outerPair.first << " " << outerPair.second.toString() << std::endl;
    }
    outputFile.close();
}

void output3(string str, long long runtime) {
    ifstream inputFile(str);
    long long valueFromFile;
    inputFile >> valueFromFile;
    inputFile.close();

    long long result = valueFromFile + runtime;

    std::ofstream outputFile(str, std::ios::trunc);
    outputFile << result;
    outputFile.close();
}

void output4(string str, unordered_map<int, double> texit) {
    std::ofstream outputFile(str);

    for (const auto& outerPair : texit) {
        outputFile << outerPair.first << " " << outerPair.second << std::endl;
    }
    outputFile.close();
}

double findAvg(vector<vector<double>>& computationMatrix, int index) {
    int sum = 0;
    int num = 0;
    for (int i = 0; i < computationMatrix[index].size(); i++) {
        if (computationMatrix[index][i] != numeric_limits<double>::infinity()) {
            sum += computationMatrix[index][i];
            num += 1;
        }
    }
    return static_cast<double>(sum) / num;
}

double findAvg_1(vector<vector<double>>& computationMatrix, int index, int proc) {
    int sum = 0;
    int num = 0;
    for (int i = 0; i < computationMatrix[index].size(); i++) {
        if (i == proc) {
            continue;
        }
        if (computationMatrix[index][i] != numeric_limits<double>::infinity()) {
            sum += computationMatrix[index][i];
            num += 1;
        }
    }
    return static_cast<double>(sum) / num;
}

bool node_can_be_processed(Graph& graph, int node) {
    for (int prednode : graph.parentSets[node]) {
        if (graph.dict[prednode].find("t") == graph.dict[prednode].end() || graph.dict[prednode].find("visit") == graph.dict[prednode].end()) {
            return false;
        }
    }
    return true;
}


ScheduleEvent ETF(unordered_map<int, vector<ScheduleEvent>>& procSchedules,
    int& offset, Graph& graph, int& node, int& proc, int& root_node,
    vector<vector<double>>& communicationMatrix, vector<vector<double>>& computationMatrix,
    unordered_map<int, NodeInfo>& dict1, int& jobId,
    unordered_map<int, tuple<int, int, vector<int>, double, double, int>>& table,
    unordered_map<int, vector<vector<double>>>& communication1,
    unordered_map<int, ScheduleEvent>& taskSchedules) {

    double ready_time = offset;
    double ready_time_t = -1;


    for (int prenode : graph.parentSets[node]) {
        if (prenode == root_node) {
            continue;
        }
        ScheduleEvent predjob = taskSchedules[prenode];
        if (predjob.proc == proc) {
            ready_time_t = predjob.end;
        }
        else {
            ready_time_t = predjob.end + graph.edgeWeights[{predjob.task, node}]/ communicationMatrix[predjob.proc][proc];
        }
        if (ready_time_t > ready_time) {
            ready_time = ready_time_t;
        }
    }
    if (node != root_node) {
        NodeInfo curr_task = getTask(node, dict1);
        if (curr_task.jobId != jobId) {
            int offset1 = node - curr_task.baseId;
            auto predecessors = curr_task.predecessors;
            for (int pre_node : predecessors) {
                auto it = std::find(graph.nodes.begin(), graph.nodes.end(), pre_node);
                if (it!= graph.nodes.end()) {
                    continue;
                }
                if (table.find(pre_node) != table.end()) {
                    auto predjob = table[pre_node];
                    if (std::get<0>(predjob) == proc) {
                        ready_time_t = std::get<3>(predjob);
                    } else if(proc != std::get<0>(table[curr_task.id]) && getTask(pre_node, dict1).id != -1){
                        ready_time_t = offset + communication1[curr_task.jobId][std::get<5>(predjob) - offset1][curr_task.baseId] / computationMatrix[std::get<0>(predjob)][proc];
                    } else {
                        ready_time_t = std::get<3>(predjob) + communication1[curr_task.jobId][std::get<5>(predjob) - offset1][curr_task.baseId] / communicationMatrix[std::get<0>(predjob)][proc];
                    }
                    if (ready_time_t > ready_time) {
                        ready_time = ready_time_t;
                    }
                }
            }
        }
    }

    int computation_time = computationMatrix[node][proc];

    auto job_list = procSchedules[proc];
    ScheduleEvent min_schedule;
    for (int idx = 0; idx < job_list.size(); idx++) {
        auto prev_job = job_list[idx];
        if (idx == 0) {
            if (prev_job.start - computation_time - ready_time > 0) {
                int job_start = ready_time;
                min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc);
                break;
            }
        }
        if (idx == job_list.size() - 1) {
            int job_start = max(ready_time, prev_job.end);
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc);
            break;
        }
        auto next_job = job_list[idx+1];
        if (next_job.start - computation_time - max(ready_time, prev_job.end) >= 0) {
            int job_start = max(ready_time, prev_job.end);
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc);
            break;
        }
    }
    if (job_list.empty()) {
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc);
    }
    return min_schedule;
}

double findReadyTime(int& node, double avgComm, int& time_offset, int& root_node, vector<vector<double>>& communicationMatrix, vector<vector<double>>& computationMatrix, unordered_map<int, vector<vector<double>>>& communication1, unordered_map<int, NodeInfo>& dict1, unordered_map<int, NodeInfo>& dict2, vector<int>& running, unordered_map<int, tuple<int, int, vector<int>, double, double, int>>& table) {
    if (node != root_node) {
        auto curr_task = getTask(node, dict1);
        int offset = curr_task.id - curr_task.baseId;
        double EST = 0;
        double avgCommunicationCost = avgComm;
        double t1 = 0;
        auto job = table[node];
        for (int pred : curr_task.predecessors) {
            auto predjob_1 = table[pred];
            double tmp = max(static_cast<double>(0), std::get<3>(predjob_1) + communication1[curr_task.jobId][std::get<5>(predjob_1) - offset][curr_task.baseId] / communicationMatrix[std::get<0>(predjob_1)][std::get<0>(job)] - time_offset);
            t1 = max(t1, tmp);
        }
        for (int prednode : curr_task.predecessors) {
            if (find(running.begin(), running.end(), prednode) == running.end()) {
                continue;
            }
            auto predjob = table[prednode];
            double temp;
            if (getTask(prednode, dict2).id != -1) {
                double sum = (communicationMatrix[node][std::get<0>(job)] + t1) + (
                            findAvg_1(computationMatrix, node, std::get<0>(job)) + (
                                communication1[curr_task.jobId][prednode - offset][
                                    node - offset] / avgCommunicationCost));
                temp = (findAvg_1(computationMatrix, node, std::get<0>(job)) + (
                            communication1[curr_task.jobId][prednode - offset][
                                node - offset] / avgCommunicationCost)) / sum * (
                               time_offset + communication1[curr_task.jobId][prednode - offset][
                           node - offset] / avgComm) + (computationMatrix[node][std::get<0>(job)] + t1) / sum * (
                               std::get<3>(predjob) + communication1[curr_task.jobId][prednode - offset][
                           node - offset] / avgComm);
            }

            else {
                temp = std::get<3>(predjob) + communication1[curr_task.jobId][prednode - offset][
                    node - offset] / avgComm;
            }
            if (temp > EST) {
                EST = temp;
            }
        }
        return EST;
    }
    return -1;
}

int main() {
    Graph graph;
    graph.readFromFile("C:/Users/32628/Desktop/dag.txt");
    vector<int> sorted_nodes;
    vector<int> overtimes = load1("C:/Users/32628/Desktop/overtime.txt");


    ifstream file("C:/Users/32628/Desktop/_self.txt");

    vector<vector<double>> communicationMatrix;
    unordered_map<int, vector<ScheduleEvent>> procSchedules;
    vector<vector<double>> computationMatrix;

    string line;
    // 读取 communication 部分
    getline(file, line); // 跳过 "communication:"
    while (getline(file, line) && line!= "communication1:") {
        vector<double> row;
        istringstream iss(line);
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        communicationMatrix.push_back(row);
    }

    unordered_map<int, vector<vector<double>>> communication1;
    int node = -1;
    vector<vector<double>> computationMatrix1;
    while (getline(file, line) && line!= "deadline_dict:") {
        if (line.length() < 5 && line != "") {
            if (node != -1) {
                communication1[node] = computationMatrix1;
                computationMatrix1.clear();
            }
            node = stoi(line);
        }
        else {
            vector<double> row;
            istringstream iss(line);
            double value;
            while (iss >> value) {
                row.push_back(value);
            }
            computationMatrix1.push_back(row);
        }

    }
    communication1[node] = computationMatrix1;

    unordered_map<int, int> deadline_dict;
    while (getline(file, line) && line!= "proc_schedules:") {
        regex pattern("(\\d+):(\\d+)");
        smatch match;
        if (std::regex_search(line, match, pattern)) {
            int n = std::stoi(match[1]);
            int deadline = std::stoi(match[2]);
            deadline_dict[n] = deadline;
        }
    }

    // 读取 proc_schedules 部分
    while (getline(file, line) && line!= "table:") {
        istringstream iss(line);
        int key;
        iss >> key;
        string str;
        getline(iss, str);
        vector<ScheduleEvent> events;
        if (str != ":[]") {
            regex eventRegex(R"(ScheduleEvent\(task=(\d+), start=(\d+|\d+\.\d+), end=(\d+|\d+\.\d+), proc=(\d+)\))");
            sregex_iterator next(str.begin(), str.end(), eventRegex);
            sregex_iterator end;

            while (next != end) {
                auto match = *next;
                ScheduleEvent event;
                event.task = stoi(match[1]);
                event.start = stod(match[2]);
                event.end = stod(match[3]);
                event.proc = stoi(match[4]);
                events.push_back(event);
                next++;
            }
        }
        procSchedules[key] = events;
    }

    unordered_map<int, tuple<int, int, vector<int>, double, double, int>> table;
    while (getline(file, line) && line!= "computation:") {
        if (line != "") {
            regex pattern("(\\d+):\\((\\d+), (\\d+), \\[(.*?)\\], (\\d+.\\d+), (\\d+.\\d+), (\\d+)\\)");
            smatch matches;
            if (regex_search(line, matches, pattern)) {
                int id = stoi(matches[1].str());
                int proc_num = stoi(matches[2].str());
                int idx = stoi(matches[3].str());
                vector<int> prenodesList;
                if (matches[4] != "") {
                    int prenode = stoi(matches[4].str());
                    prenodesList.push_back(prenode);
                }
                double end = stod(matches[5].str());
                double start = stod(matches[6].str());

                table[id] = make_tuple(proc_num, idx, prenodesList, end, start, id);
            }
        }
    }

    // 读取 computation 部分
    while (getline(file, line) && line != "texit:") {
        vector<double> numbers;

        // 正则表达式匹配数字或inf
        regex pattern(R"((\s*(\d+\.\d+|\d+|inf)\s*))");

        // 使用sregex_iterator遍历所有匹配项
        sregex_iterator it(begin(line), end(line), pattern);
        sregex_iterator end;

        for (; it != end; ++it) {
            string match = it->str();
            // 去除前后的空格
            match.erase(remove_if(match.begin(), match.end(), ::isspace), match.end());
            if (match == "inf") {
                numbers.push_back(numeric_limits<double>::infinity());
            } else {
                numbers.push_back(stod(match));
            }
        }
        computationMatrix.push_back(numbers);
    }

    unordered_map<int, double> texit;
    while (getline(file, line) && line!= "running:") {
        if (line != "") {
            regex pattern("(\\d+):([\\d\\.]+)");
            smatch matches;
            if (regex_search(line, matches, pattern)) {
                int firstNumber = stoi(matches[1]);
                double secondNumber = stod(matches[2]);
                texit[firstNumber] = secondNumber;
            }
        }
    }

    getline(file, line);
    vector<int> running;
    if (line.length() > 2) {
        stringstream ss(line.substr(1, line.length() - 2));
        string item;
        while (getline(ss, item, ',')) {
            if (!item.empty()) {
                int num = stoi(item);
                running.push_back(num);
            }
        }
    }

    while (line != "root_node:") {
        getline(file, line);
    }



    // 读取 computation 部分
    getline(file, line);
    int root_node = stoi(line);

    while (line != "terminal_node:") {
        getline(file, line);
    }
    getline(file, line);
    int terminal_node = stoi(line);

    while (line != "offset:") {
        getline(file, line);
    }
    getline(file, line);
    int offset = stoi(line);

    while (line != "jobId:") {
        getline(file, line);
    }

    getline(file, line);
    int jobId = stoi(line);

    while (line != "deadline:") {
        getline(file, line);
    }
    getline(file, line);
    int deadline = stoi(line);

    while (line != "avgComm:") {
        getline(file, line);
    }
    getline(file, line);
    int avgComm = stod(line);

    file.close();

    unordered_map<int, NodeInfo> dict1 = getDict("C:/Users/32628/Desktop/dict1.txt");
    unordered_map<int, NodeInfo> dict2 = getDict("C:/Users/32628/Desktop/dict2.txt");

    ifstream file2("C:/Users/32628/Desktop/task_schedules.txt");

    string content((istreambuf_iterator<char>(file2)), istreambuf_iterator<char>());

    regex pattern("(\\d+):(ScheduleEvent\\(task=(\\d+), start=(\\d+(?:\\.\\d+)?), end=(\\d+(?:\\.\\d+)?), proc=(\\d+)\\))?");
    sregex_iterator it(content.begin(), content.end(), pattern);
    sregex_iterator end;

    unordered_map<int, ScheduleEvent> taskSchedules;

    while (it!= end) {
        int task = stoi(it->str(1));
        if (it->str(2).empty()) {
            taskSchedules[task] = ScheduleEvent(-1, 0, 0, -1);
        } else {
            int taskNum = stoi(it->str(3));
            double start = stod(it->str(4));
            double endValue = stod(it->str(5));
            int proc = stoi(it->str(6));
            taskSchedules[task] = ScheduleEvent(taskNum, start, endValue, proc);
        }
        ++it;
    }
    long long duration1 = 0;
    long long duration2 = 0;
    auto s1 = high_resolution_clock::now();
    graph.dict[root_node]["t"] = findAvg(computationMatrix, root_node);
    graph.dict[root_node]["t1"] = 0;
    graph.dict[root_node]["visit"] = 0;

    deque<int> visit_queue;
    visit_queue.push_back(root_node);
    while (!visit_queue.empty()) {
        int node = visit_queue.front();
        visit_queue.pop_front();
        while (!node_can_be_processed(graph, node)) {
            int n1 = visit_queue.front();
            visit_queue.pop_front();
            visit_queue.push_back(node);
            node = n1;
        }
        graph.dict[node]["visit"] = 0;
        double temp_1 = 0;
        if (graph.dict[node].find("sd") != graph.dict[node].end()) {
            if (find(overtimes.begin(), overtimes.end(), node) != overtimes.end()) {
                for (int succ : graph.successors[node]) {
                    if (find(visit_queue.begin(), visit_queue.end(), succ) == visit_queue.end()) {
                        visit_queue.push_back(succ);
                    }
                }
                continue;
            }
            double EST = findReadyTime(node, avgComm, offset, root_node, communicationMatrix, computationMatrix, communication1, dict1, dict2, running, table);
            double DRT = max(EST - offset, double(0));
            double maxDept = 0;
            for (int prednode : graph.parentSets[node]) {
                double temp = graph.dict[prednode]["t"] + graph.edgeWeights[{prednode,node}];
                maxDept = max(maxDept, temp);
            }
            temp_1 = max(DRT, maxDept);
        }
        else {
            for (int prednode : graph.parentSets[node]) {
                double temp = graph.dict[prednode]["t"] + graph.avgWeights[{prednode,node}];
                temp_1 = max(temp_1, temp);
            }
        }
        graph.dict[node]["t1"] = temp_1;
        graph.dict[node]["t"] = temp_1 + findAvg(computationMatrix, node);
        if (graph.parentSets[terminal_node].find(node) != graph.parentSets[terminal_node].end()) {
            if (graph.dict[node].find("sd") == graph.dict[node].end()) {
                if (texit.find(jobId) != texit.end()) {
                    texit[jobId] = max(graph.dict[node]["t"], texit[jobId]);
                } else {
                    texit[jobId] = graph.dict[node]["t"];
                }
            }
        }
        for (int succ : graph.successors[node]) {
            if (find(visit_queue.begin(), visit_queue.end(), succ) == visit_queue.end()) {
                visit_queue.push_back(succ);
            }
        }
    }
    for (int node : graph.nodes) {
        graph.dict[node].erase("visit");
        if (graph.dict[node].find("sd") == graph.dict[node].end()) {
            graph.dict[node]["sd"] = (graph.dict[node]["t"] / texit[jobId] * deadline + graph.dict[node]["t1"]) / 2;
        } else {
            auto node_info = getTask(node, dict1);
            if (deadline_dict[node_info.jobId] > offset) {
                graph.dict[node]["sd"] = (graph.dict[node]["t"] / texit[node_info.jobId] * (
                            deadline_dict[node_info.jobId] - offset) + graph.dict[node]["t1"]) / 2;
            }
        }
    }
    auto e1 = high_resolution_clock::now();
    // 计算持续时间
    duration1 += duration_cast<microseconds>(e1 - s1).count();
    sort(graph.nodes.begin(), graph.nodes.end(), [&graph](int a, int b) {
            if (graph.dict[a]["sd"] < graph.dict[b]["sd"]) {
                return true;
            } else if (graph.dict[a]["sd"] > graph.dict[b]["sd"]){
                return false;
            } else {
                if (a < b) {
                    return true;
                } else {
                    return false;
                }
            }
    });
    s1 = high_resolution_clock::now();
    for (int node : graph.nodes) {
        auto it = std::find(overtimes.begin(), overtimes.end(), node);
        if (it != overtimes.end()) {
            continue;
        }
        if (node == root_node || node == terminal_node) {
            continue;
        }
        ScheduleEvent minTaskSchedule = ScheduleEvent(node, numeric_limits<double>::infinity(), numeric_limits<double>::infinity(), -1);
        for (int proc = 0; proc < communicationMatrix.size() - 1; proc++) {
            if (computationMatrix[node][proc] == numeric_limits<double>::infinity()) {
                continue;
            }
            ScheduleEvent taskschedule = ETF(procSchedules, offset, graph, node, proc, root_node,communicationMatrix, computationMatrix, dict1,  jobId, table, communication1, taskSchedules);
            if (taskschedule.end < minTaskSchedule.end) {
                minTaskSchedule = taskschedule;
            }
        }
        taskSchedules[node] = minTaskSchedule;
        procSchedules[minTaskSchedule.proc].push_back(minTaskSchedule);
        sort(procSchedules[minTaskSchedule.proc].begin(), procSchedules[minTaskSchedule.proc].end(), [](const ScheduleEvent& a, const ScheduleEvent& b) {
            return a.end < b.end;
        });
    }
    e1 = high_resolution_clock::now();
    // 计算持续时间
    duration1 += duration_cast<microseconds>(e1 - s1).count();

    unordered_map<int, tuple<int, int, vector<int>, double, double, int>> dict_output;
    for (auto pair : procSchedules) {
        int proc_num = pair.first;
        auto proc_tasks  = pair.second;
        for (size_t idx = 0; idx < proc_tasks.size(); ++idx) {
            auto task = proc_tasks[idx];
            if (idx > 0 && (proc_tasks[idx - 1].end - proc_tasks[idx - 1].start > 0)) {
                dict_output[task.task] = {proc_num, static_cast<int>(idx), {proc_tasks[idx - 1].task}, task.end, task.start, task.task};
            } else {
                dict_output[task.task] = {proc_num, static_cast<int>(idx), {}, task.end, task.start, task.task};
            }
        }
    }

    output("C:/Users/32628/Desktop/dictOutput.txt", dict_output);
    output1("C:/Users/32628/Desktop/procSchedules.txt", procSchedules);
    output2("C:/Users/32628/Desktop/taskSchedules.txt", taskSchedules);
    output3("C:/Users/32628/Desktop/runtime.txt", duration1);
    output4("C:/Users/32628/Desktop/texit1.txt", texit);
    return 0;
}
