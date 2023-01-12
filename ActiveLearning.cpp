#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>

using namespace std;

vector<vector<double>> loadcsv(string csv_file)
{
    ifstream in_file(csv_file);

    vector<vector<double>> data;

    // Read the data from the file
    string line;
    getline(in_file, line);
    while (getline(in_file, line))
    {
        // Create a string stream from the line
        istringstream iss(line);

        // Define a vector for the current row
        vector<double> row;

        // Read the values from the line
        string value;
        while (getline(iss, value, ','))
        {
            // Convert the value to a double and add it to the row vector
            row.push_back(stod(value));
        }

        // Add the row vector to the 2 dimensional vector
        data.push_back(row);
    }

    // Close the input file
    in_file.close();
    return data;
}

vector<vector<vector<double>>> loadcsv_QBC_KLDiv(string csv_file, int models, int classes)
{
    ifstream in_file(csv_file);

    vector<vector<vector<double>>> data;

    // Read the data from the file
    string line;
    getline(in_file, line);
    int row_no = 0;
    while (getline(in_file, line))
    {

        // Create a string stream from the line
        istringstream iss(line);

        // Define a vector for the current row

        // Read the values from the line
        string value;
        vector<vector<double>> models_no;
        for (int i = 0; i < models + 1; i++)
        {
            vector<double> row;
            if (i == 0)
            {
                getline(iss, value, ',');
                // Convert the value to a double and add it to the row vector
                row.push_back(stod(value));
                // cout << stod(value) << endl;
            }
            else
            {
                for (int j = 0; j < classes; j++)
                {
                    getline(iss, value, ',');

                    // Convert the value to a double and add it to the row vector
                    row.push_back(stod(value));
                }
            }
            models_no.push_back(row);
        }
        data.push_back(models_no);
        row_no++;
    }

    // Close the input file
    in_file.close();
    // cout << data[0].size();
    return data;
}

void least_confidence(vector<vector<double>> data)
{
    vector<double> res_lc;
    vector<pair<double, double>> toptwo;

    for (int i = 0; i < data.size(); i++)
    {

        vector<double> vec;
        for (int j = 1; j < data[0].size(); j++)
        {
            vec.push_back(data[i][j]);
        }

        sort(vec.begin(), vec.end());
        // cout << vec[vec.size() - 1].second << ", " << vec[vec.size() - 2].second;
        // cout << endl;
        double prob = vec[vec.size() - 1];
        toptwo.push_back(make_pair(prob, data[i][0]));
    }
    sort(toptwo.begin(), toptwo.end());

    for (int i = 0; i < toptwo.size(); i++)
    {
        // cout << toptwo[i].first << endl;
        res_lc.push_back(toptwo[i].second);
    }

    ofstream f1("data_lc.csv");
    if (!f1.is_open())
    {
        cerr << "Error: Unable to open file : f1." << endl;
    }

    for (int i = 0; i < res_lc.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_lc[i];
        string line = ss.str();
        f1 << line << endl;
    }
    f1.close();

    system("py ML_Assignment2_copy.py retrain data_lc.csv");
}

void smallest_margin(vector<vector<double>> data)
{
    vector<double> res_sm;
    vector<pair<double, double>> toptwo;

    for (int i = 0; i < data.size(); i++)
    {

        vector<double> vec;
        for (int j = 1; j < data[0].size(); j++)
        {
            vec.push_back(data[i][j]);
        }

        sort(vec.begin(), vec.end());
        // cout << vec[vec.size() - 1] << ", " << vec[vec.size() - 2];
        // cout << endl;
        double prob = vec[vec.size() - 1] - vec[vec.size() - 2];
        toptwo.push_back(make_pair(prob, data[i][0]));
    }
    sort(toptwo.begin(), toptwo.end());

    for (int i = 0; i < toptwo.size(); i++)
    {
        // cout << toptwo[i].first << " " << toptwo[i].second << endl;
        res_sm.push_back(toptwo[i].second);
    }
    ofstream f2("data_sm.csv");
    if (!f2.is_open())
    {
        cerr << "Error: Unable to open file : f2." << endl;
    }
    for (int i = 0; i < res_sm.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_sm[i];
        string line = ss.str();
        f2 << line << endl;
    }
    f2.close();
    system("py ML_Assignment2_copy.py retrain data_sm.csv");
}

void largest_margin(vector<vector<double>> data)
{
    vector<double> res_lm;
    vector<pair<double, double>> diff;

    for (int i = 0; i < data.size(); i++)
    {

        vector<double> vec;
        for (int j = 1; j < data[0].size(); j++)
        {
            vec.push_back(data[i][j]);
        }

        sort(vec.begin(), vec.end());
        // cout << vec[vec.size() - 1].second << ", " << vec[vec.size() - 2].second;
        // cout << endl;
        double prob = vec[vec.size() - 1] - vec[0];
        diff.push_back(make_pair(prob, data[i][0]));
    }
    sort(diff.begin(), diff.end());

    for (int i = 0; i < diff.size(); i++)
    {
        // cout << toptwo[i].first << " " << toptwo[i].second << endl;
        res_lm.push_back(diff[i].second);
    }
    ofstream f3("data_lm.csv");
    if (!f3.is_open())
    {
        cerr << "Error: Unable to open file : f3." << endl;
    }
    for (int i = 0; i < res_lm.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_lm[i];
        string line = ss.str();
        f3 << line << endl;
    }
    f3.close();
    system("py ML_Assignment2_copy.py retrain data_lm.csv");
}

double KL_Div(vector<double> p, vector<double> q)
{

    double d_KL = 0;
    for (int i = 0; i < p.size(); i++)
    {
        if (q[i] == 0)
            q[i] = 1.0e-45;
        if (p[i] != 0)
            d_KL += p[i] * log((p[i] / q[i]));
    }
    return d_KL;
}

void entropy_sampling(vector<vector<double>> data)
{
    int num_rows = data.size();
    vector<pair<double, int>> entropy(num_rows);
    for (int i = 0; i < num_rows; i++)
    {
        double value = 0.0;
        for (int j = 1; j <= 10; j++)
        {
            if (data[i][j] != 0)
                value += (data[i][j] * log(data[i][j]));
            else
                value += (0);
        }
        value = value * -1;
        entropy[i] = make_pair(value, data[i][0]);
    }
    sort(entropy.rbegin(), entropy.rend());
    vector<int> res_ent(num_rows);
    for (int i = 0; i < num_rows; i++)
        res_ent[i] = entropy[i].second;
    ofstream f4("data_entropy.csv");
    if (!f4.is_open())
    {
        cerr << "Error: Unable to open file : f4." << endl;
    }
    for (int i = 0; i < res_ent.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_ent[i];
        string line = ss.str();
        f4 << line << endl;
    }
    f4.close();
    system("py ML_Assignment2_copy.py retrain data_entropy.csv");
}

void QBC_vote_entropy(vector<vector<double>> data, int models, int classes)
{
    vector<pair<double, double>> res_vote;

    int num_rows = data.size();
    for (int i = 0; i < num_rows; i++)
    {
        double value = 0.0;
        for (int j = 1; j <= classes; j++)
        {
            if (data[i][j] != 0.0)
                value += ((data[i][j] / models) * (log(data[i][j] / models)));

            // cout << value << endl;
        }
        res_vote.push_back(make_pair(value, data[i][0]));
        // cout << value << endl;
    }
    sort(res_vote.begin(), res_vote.end());
    vector<int> res_vote_entropy(num_rows);
    for (int i = 0; i < num_rows; i++)
    {
        res_vote_entropy[i] = res_vote[i].second;
        // cout << res[i].first << endl;
    }

    ofstream f5("res_VoteEntropy.csv");
    if (!f5.is_open())
    {
        cerr << "Error: Unable to open file : f5." << endl;
    }
    for (int i = 0; i < res_vote_entropy.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_vote_entropy[i];
        string line = ss.str();
        f5 << line << endl;
    }
    f5.close();

    int vote_ent_VS = 0;
    for (int i = 0; i < res_vote.size(); i++)
    {
        if (res_vote[i].first != 0)
        {
            vote_ent_VS++;
        }
    }
     system("py qbcEntropy_copy.py retrain res_VoteEntropy.csv");

    cout << "VERSION SPACE (VOTE ENTROPY): " << vote_ent_VS << endl;
}

void QBC_KLdiv(vector<vector<vector<double>>> data, int models, int classes)
{
    vector<pair<double, double>> res_KLdiv;

    vector<pair<int, int>> pairs;
    for (int i = 1; i <= models; i++)
    {
        for (int j = 1; j <= models; j++)
        {
            if (i != j)
            {
                pairs.push_back({i, j});
                // cout << i << " " << j << endl;
            }
        }
    }
    // cout << pairs[0].second;

    vector<double> res_KL;
    for (int i = 0; i < data.size(); i++)
    {
        // double kl_classes = 0;
        double sum = 0;
        for (int j = 0; j < pairs.size(); j++) // models
        {
            // double sum = 0;
            // cout << pairs[j].first();
            sum += KL_Div(data[i][pairs[j].first], data[i][pairs[j].second]);
            //     kl_classes += sum;
        }
        sum /= pairs.size();
        res_KLdiv.push_back({sum, data[i][0][0]});
    }

    sort(res_KLdiv.begin(), res_KLdiv.end());

    for (int i = 0; i < res_KLdiv.size(); i++)
    {
        // cout << res_KLdiv[i].second << " ";
        // cout << res_KLdiv[i].first << endl;
        res_KL.push_back(res_KLdiv[i].second);
    }

    ofstream f6("data_kl_diver.csv");

    if (!f6.is_open())
    {
        cerr << "Error: Unable to open file : f6." << endl;
    }
    for (int i = 0; i < res_KL.size(); i++)
    {
        // cout << res[i] << endl;
        stringstream ss;
        ss << res_KL[i];
        string line = ss.str();
        f6 << line << endl;
    }
    f6.close();

     system("py qbcEntropy_copy.py retrain data_kl_diver.csv");


    int kl_div_VS = 0;
}

void K_means_accuracy(vector<vector<double>> data)
{
    double data_size = data.size();
    double correct_classification = 0;
    for (int i = 0; i < data_size; i++)
    {
        if (data[i][1] == data[i][2])
        {
            correct_classification++;
        }
    }
    double res = (correct_classification / data_size);
    // cout << correct_classification << endl;
    // cout << data_size;
    cout << "K-MEANS ACCURACY : " << res << endl;
}

int main()
{

    system("py ML_Assignment2_copy.py new nofile");
    
    for(int i=1; i<=4; i++){
        vector<vector<double>> data = loadcsv("predic.csv");
        least_confidence(data);
    }

    system("py ML_Assignment2_copy.py new nofile");

    for(int i=1; i<=4; i++){
        vector<vector<double>> data = loadcsv("predic.csv");
        smallest_margin(data);
    }

    system("py ML_Assignment2_copy.py new nofile");

    for(int i=1; i<=4; i++){
        vector<vector<double>> data = loadcsv("predic.csv");
        largest_margin(data);
    }

    system("py ML_Assignment2_copy.py new nofile");

    for(int i=1; i<=4; i++){
        vector<vector<double>> data = loadcsv("predic.csv");
        entropy_sampling(data);
    }

    system("py qbcEntropy_copy.py new nofile");

    vector<vector<double>> data_votes = loadcsv("votes.csv");
    QBC_vote_entropy(data_votes, 5, 6);
        
    vector<vector<vector<double>>>data_QBC = loadcsv_QBC_KLDiv("logs.csv", 5, 6);
    QBC_KLdiv(data_QBC, 5, 6);
    
    // Version_space();

    vector<vector<double>>
        data_kmeans = loadcsv("k_cluster.csv");
    K_means_accuracy(data_kmeans);
    // cout << k_means_accuracy;
}