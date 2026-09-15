// STARGUARD AI — telemetry_engine
//
// Высокопроизводительный модуль потоковой обработки телеметрии СМАРТ ТМ/АВР
// (температура токоведущих частей, ток нагрузки). Задача: считать
// скользящую статистику, детектировать устойчивые аномалии (z-score + EWMA)
// и дать линейную оценку остаточного ресурса (RUL) до достижения
// критического порога — без обращения к Python/облаку, пригодно для работы
// на edge-устройстве рядом со шкафом.
//
// Протокол ввода (stdin):  <day>,<temperature_c>,<current_a>\n  на строку
// Протокол вывода (stdout): один JSON-объект с результатом анализа.
//
// Режим бенчмарка: telemetry_engine --bench N
//   Генерирует N синтетических точек в памяти и печатает достигнутую
//   пропускную способность (точек/сек), чтобы показать, зачем этот модуль
//   написан на C++, а не на Python.

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>

struct Sample {
    int day;
    double temp;
    double current;
};

struct RollingStats {
    std::deque<double> window;
    double sum = 0.0;
    double sumsq = 0.0;
    size_t cap;

    explicit RollingStats(size_t capacity) : cap(capacity) {}

    void push(double v) {
        window.push_back(v);
        sum += v;
        sumsq += v * v;
        if (window.size() > cap) {
            double old = window.front();
            window.pop_front();
            sum -= old;
            sumsq -= old * old;
        }
    }

    double mean() const {
        return window.empty() ? 0.0 : sum / static_cast<double>(window.size());
    }

    double stddev() const {
        if (window.size() < 2) return 1e-6;
        double m = mean();
        double var = sumsq / static_cast<double>(window.size()) - m * m;
        return std::sqrt(std::max(var, 1e-6));
    }
};

// Наименьших квадратов наклон (slope) по последним точкам окна.
double linreg_slope(const std::deque<double>& ys) {
    size_t n = ys.size();
    if (n < 3) return 0.0;
    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double y = ys[i];
        sx += x; sy += y; sxy += x * y; sxx += x * x;
    }
    double denom = (static_cast<double>(n) * sxx - sx * sx);
    if (std::abs(denom) < 1e-9) return 0.0;
    return (static_cast<double>(n) * sxy - sx * sy) / denom;
}

struct AnalysisResult {
    std::vector<double> zscores;
    std::vector<double> smoothed;
    std::string status = "OK";
    int anomaly_start_day = -1;
    double rul_days_estimate = -1.0;
    double throughput_points_per_sec = 0.0;
};

AnalysisResult analyze(const std::vector<Sample>& samples, size_t window_size = 14,
                        double ewma_alpha = 0.1, double anomaly_threshold = 3.0,
                        int sustained_days_required = 10, size_t baseline_window = 20) {
    AnalysisResult res;

    // Важно: используем ФИКСИРОВАННУЮ базовую линию, снятую при пуско-наладке
    // (первые baseline_window дней), а не скользящее окно. Скользящее окно
    // "подстраивается" под медленный дрейф и перестаёт видеть его как аномалию —
    // это ключевая слабость наивных детекторов, которую и призвана закрыть
    // нейросеть раннего предупреждения (см. predictive_model.py).
    RollingStats baseline(baseline_window);
    size_t n_for_baseline = std::min(baseline_window, samples.size());
    for (size_t i = 0; i < n_for_baseline; ++i) baseline.push(samples[i].temp);
    double baseline_mean = baseline.mean();
    double baseline_std = baseline.stddev();

    double ewma = 0.0;
    bool ewma_init = false;
    int consecutive_anomalous = 0;
    int first_sustained_day = -1;

    for (const auto& s : samples) {
        double z = (s.temp - baseline_mean) / baseline_std;
        res.zscores.push_back(z);

        if (!ewma_init) { ewma = z; ewma_init = true; }
        else { ewma = ewma_alpha * z + (1 - ewma_alpha) * ewma; }
        res.smoothed.push_back(ewma);

        if (ewma > anomaly_threshold) {
            consecutive_anomalous++;
            if (consecutive_anomalous >= sustained_days_required && first_sustained_day == -1) {
                first_sustained_day = s.day - sustained_days_required + 1;
            }
        } else {
            consecutive_anomalous = 0;
        }
    }

    res.anomaly_start_day = first_sustained_day;
    if (first_sustained_day >= 0) {
        res.status = "CRITICAL";
    } else {
        // мягкое предупреждение, если последние точки уже заметно выше нормы
        if (!res.smoothed.empty() && res.smoothed.back() > anomaly_threshold * 0.6) {
            res.status = "WARNING";
        }
    }

    // Оценка RUL: тренд по последнему окну температуры, критический порог = базовая
    // линия пуско-наладки + 15C (типовой допуск нагрева для НКУ).
    if (samples.size() > window_size * 2) {
        double critical_level = baseline_mean + 8.0;

        std::deque<double> recent;
        size_t start = samples.size() > window_size ? samples.size() - window_size : 0;
        for (size_t i = start; i < samples.size(); ++i) recent.push_back(samples[i].temp);
        double slope = linreg_slope(recent); // °C / день
        double current_val = samples.back().temp;

        if (slope > 0.01 && current_val < critical_level) {
            res.rul_days_estimate = (critical_level - current_val) / slope;
        } else if (current_val >= critical_level) {
            res.rul_days_estimate = 0.0;
        } else {
            res.rul_days_estimate = -1.0; // нет устойчивого тренда деградации
        }
    }

    return res;
}

std::string to_json(const AnalysisResult& r) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "{";
    out << "\"status\":\"" << r.status << "\",";
    out << "\"anomaly_start_day\":" << r.anomaly_start_day << ",";
    out << "\"rul_days_estimate\":" << r.rul_days_estimate << ",";
    out << "\"smoothed_zscore\":[";
    for (size_t i = 0; i < r.smoothed.size(); ++i) {
        out << r.smoothed[i];
        if (i + 1 < r.smoothed.size()) out << ",";
    }
    out << "]";
    out << "}";
    return out.str();
}

void run_stream_mode() {
    std::vector<Sample> samples;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        Sample s{};
        std::getline(ss, tok, ','); s.day = std::stoi(tok);
        std::getline(ss, tok, ','); s.temp = std::stod(tok);
        std::getline(ss, tok, ','); s.current = std::stod(tok);
        samples.push_back(s);
    }
    auto res = analyze(samples);
    std::cout << to_json(res) << std::endl;
}

void run_benchmark(long n) {
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.6);
    std::vector<Sample> samples;
    samples.reserve(n);
    double t = 40.0;
    for (long i = 0; i < n; ++i) {
        t += noise(gen) * 0.05;
        samples.push_back({static_cast<int>(i), t, 300.0 + noise(gen) * 10});
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = analyze(samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    double throughput = static_cast<double>(n) / std::max(seconds, 1e-9);

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "{\"benchmark_points\":" << n << ",";
    out << "\"elapsed_seconds\":" << seconds << ",";
    out << "\"throughput_points_per_sec\":" << throughput << ",";
    out << "\"status\":\"" << res.status << "\"}";
    std::cout << out.str() << std::endl;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        long n = std::stol(argv[2]);
        run_benchmark(n);
        return 0;
    }
    run_stream_mode();
    return 0;
}
