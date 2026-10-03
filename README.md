# Hourly Load Forecasting for PG&E Energy Analytics

## Overview

This project explores hourly electricity load forecasting using the dataset from the **IISE PG&E Energy Analytics Challenge 2025**.

The objective is to forecast hourly electricity load from **weather, calendar, and time-based features** while avoiding future load information that would be unavailable in a static full-year forecasting setting.

The forecasting pipeline combines **weather feature aggregation, PLS dimensionality reduction, calendar features, Fourier time encoding, weather lags and deltas, and XGBoost regression**. Hyperparameters are optimized with **Optuna**, while chronological validation and multi-seed experiments are used to evaluate model performance and stability.

The project also includes **feature ablation experiments** to investigate the contribution of different feature groups and an additional comparison with historical-load features in a day-ahead forecasting setting.

## Method

### 1. Data & Problem Formulation

The project uses the dataset provided by the **IISE PG&E Energy Analytics Challenge 2025**. The raw data consists of two Excel files: `training.xlsx` and `testing.xlsx`. Both files share the same schema, while the `Load` column is available only in the training data and is left empty in the testing data.

| Feature Group    | Variables                      |
| ---------------- | ------------------------------ |
| Time             | `Year`, `Month`, `Day`, `Hour` |
| Target           | `Load`                         |
| Temperature      | `Site-1 Temp` – `Site-5 Temp`  |
| Solar Irradiance | `Site-1 GHI` – `Site-5 GHI`    |

The forecasting task is formulated as a supervised regression problem, where the objective is to predict hourly electricity load from the available time and weather information:

$$
Load_t = f(X_t)
$$

where \(X_t\) represents the time and weather features associated with hour \(t\).

The raw dataset is organized into three consecutive years. **Year 1 and Year 2 are available in the training data**, while **Year 3 corresponds to the testing period**, for which the ground-truth `Load` values are not provided.

To preserve the temporal structure and avoid temporal leakage, the modeling pipeline uses the following chronological protocol:

* **Year 1:** time-series cross-validation and hyperparameter optimization.
* **Year 2:** hold-out validation.
* **Year 1 + Year 2:** final model training after model selection.
* **Year 3:** final forecasting on the testing data.

### 2. EDA

The exploratory data analysis (EDA) was conducted to understand the temporal, spatial, and statistical characteristics of electricity load and weather variables. The main objective is not only to describe the dataset, but also to identify patterns that motivate the subsequent feature engineering and modeling decisions.

#### 2.1 Statistical Overview

1. Load Demand

- **Yearly Trend:** The average load remains relatively stable across the two years (~2,162 MW in Year 1 vs. ~2,145 MW in Year 2). However, the variance in Year 1 is significantly higher (216,944 vs. 165,227), as reflected by the Peak Load reaching **4,397 MW** in Year 1, compared to only **3,808 MW** in Year 2.

- **Distribution Shape:**
  - **Skewness > 0** (1.18 in Year 1 and 0.67 in Year 2) and **Kurtosis > 0** (2.10 in Year 1 and 0.84 in Year 2): The load distribution is clearly right-skewed and exhibits heavy tails.
  - *Interpretation:* Electricity demand is primarily concentrated around the average level (~2,000–2,100 MW), but occasional extreme spikes in demand occur with relatively low frequency.

2. Temperature (Site 1 to Site 5)

- **Spatial Homogeneity:**
  - The average temperatures across the five sites are highly similar, ranging from approximately **16.5°C to 18.3°C**.
  - **Site 3 is the most anomalous site (Outlier Site):** It exhibits the widest range of extreme temperatures (Min as low as -0.5°C in Year 1 and Max as high as 43.0°C in Year 1) and the highest variance (~49–54). The other sites (Site 1, 2, 4, and 5) show much smoother variations, with variances ranging from approximately ~18–34.

- **Distribution Shape:**
  - **Skewness close to 0** (-0.02 to 0.45) and **negative Kurtosis** (-0.45 to 0.02): Temperature approximately follows a normal or slightly flat (platykurtic) distribution, without the heavy tails observed in the Load distribution.

3. Solar Radiation (GHI - Global Horizontal Irradiance)

- **High Spatial Homogeneity:**
  - All five sites have highly similar mean values (~218–228 W/m²), maximum values (~1,024–1,049 W/m²), and standard deviations.

- **Distribution Characteristics Due to Nighttime Data:**
  - **Median = 11–12 W/m²** and **Min = 0:** Due to the day/night cycle (GHI = 0 during nighttime), the median is pulled down close to zero.
  - **Skewness > 1** (~1.07–1.12): The distribution is strongly right-skewed because irradiance remains very low for most of the day (during nighttime, early morning, and late afternoon), with a sharp increase occurring only during a few hours around midday.

4. Data Stability Across the Two Years (Data Drift)

- **GHI and Temperature:** The Mean, Median, Skewness, and Kurtosis values of GHI and Temperature are nearly identical between Year 1 and Year 2. This indicates that there is **no severe Covariate Shift** in the weather-related variables.

- **Load:** There is a decrease in variability in Year 2 (with a lower Peak Load). If Year 1 is used as the Training Set and Year 2 as the Validation Set, it is important to note that the model trained on Year 1 may become overly sensitive to extreme peak values (**overfitting to peaks**).

| **Year** | **Statistic** | **Load** | **Temperature - Site 1** | **Temperature - Site 2** | **Temperature - Site 3** | **Temperature - Site 4** | **Temperature - Site 5** | **GHI - Site 1** | **GHI - Site 2** | **GHI - Site 3** | **GHI - Site 4** | **GHI - Site 5** |
| -------: | :------------ | -------: | ------------------------: | ------------------------: | ------------------------: | ------------------------: | ------------------------: | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: |
| **1** | Mean | 2162.82 | 17.31 | 17.17 | 18.27 | 17.72 | 17.60 | 225.80 | 222.47 | 226.91 | 227.06 | 228.39 |
| **1** | Variance | 216944.79 | 23.65 | 20.35 | 54.16 | 22.59 | 34.06 | 93137.48 | 90908.02 | 94454.84 | 94023.73 | 94894.82 |
| **1** | Median | 2072.00 | 17.50 | 17.40 | 17.30 | 17.80 | 17.30 | 12.00 | 12.00 | 12.00 | 12.00 | 13.00 |
| **1** | Min | 1101.00 | 1.90 | 2.90 | -0.50 | 2.60 | 0.90 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **1** | Max | 4397.00 | 36.60 | 31.90 | 43.00 | 36.60 | 39.70 | 1037.00 | 1028.00 | 1041.00 | 1047.00 | 1049.00 |
| **1** | Skewness | 1.18 | 0.09 | -0.00 | 0.45 | 0.20 | 0.42 | 1.07 | 1.09 | 1.09 | 1.08 | 1.08 |
| **1** | Kurtosis | 2.10 | -0.14 | -0.33 | -0.26 | -0.02 | 0.02 | -0.25 | -0.19 | -0.20 | -0.23 | -0.23 |
| **2** | Mean | 2145.42 | 16.72 | 16.47 | 17.80 | 17.06 | 16.84 | 221.24 | 218.81 | 225.35 | 223.31 | 225.08 |
| **2** | Variance | 165227.38 | 20.09 | 17.45 | 49.22 | 18.96 | 28.81 | 91016.81 | 89239.26 | 93875.32 | 92757.19 | 93671.90 |
| **2** | Median | 2096.00 | 16.60 | 16.40 | 17.20 | 17.00 | 16.60 | 12.00 | 11.00 | 11.00 | 11.00 | 11.00 |
| **2** | Min | 1027.00 | 4.60 | 4.50 | 0.20 | 3.90 | 2.90 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **2** | Max | 3808.00 | 30.10 | 30.20 | 38.80 | 31.20 | 33.60 | 1031.00 | 1024.00 | 1035.00 | 1042.00 | 1045.00 |
| **2** | Skewness | 0.67 | 0.06 | -0.02 | 0.37 | 0.11 | 0.27 | 1.11 | 1.12 | 1.09 | 1.11 | 1.10 |
| **2** | Kurtosis | 0.84 | -0.36 | -0.45 | -0.45 | -0.30 | -0.34 | -0.16 | -0.13 | -0.21 | -0.17 | -0.18 |

#### 2.2 Seasonal Load Patterns

The load exhibits a strong seasonal structure with two major high-demand periods.

- **Summer peak:** Load increases substantially from approximately June to September, with the highest levels generally occurring during July–September. This pattern is consistent with increased cooling demand during warmer periods.
- **Winter increase:** Load increases again around December–January, although the magnitude is generally lower than the summer peak.
- **Shoulder months:** March–May and October–November generally correspond to lower demand periods.
- **Peak demand:** A large number of high-load observations occur during August–October, with extreme values reaching approximately **3,500–4,400 MW**, particularly in Year 1.

The seasonal load profiles are highly consistent between Year 1 and Year 2. However, Year 2 shows a noticeable compression of peak demand, particularly during the late-summer period.

This strong and repeatable seasonal structure motivates the use of explicit **seasonal and periodic time features**, rather than relying only on the raw `Month` variable.

<p align="center">
  <img src="report\Load Distribution by Month.png" width="800"/>
</p>

<p align="center">
  <em>Load Distribution by Month (Year 1 & 2)</em>
</p>

#### 2.3 Daily and Weekly Load Patterns

The hourly load profiles exhibit a clear daily cycle.

- Load reaches its lowest baseline around **03:00–04:00**.
- A secondary **morning peak** occurs around **07:00–08:00**.
- The dominant **evening peak** occurs around **18:00–20:00**.
- A smaller midday decline can be observed around **11:00–14:00**, particularly during lower-demand months.

The load also exhibits a strong weekly structure. Although all days follow a similar daily pattern, weekend behavior differs slightly from weekdays, particularly during transitional months. During the summer peak period, however, weekend demand can remain relatively high.

The combination of daily and weekly periodicity suggests that the model should explicitly represent:

- daily periodicity,
- weekly periodicity,
- day-of-week effects,
- and interactions between time and weather conditions.

This motivates the use of **Fourier time features** together with calendar features such as `dow` and `is_weekend`.

<p align="center">
  <img src="report\Daily_Load.png" width="800"/>
</p>

<p align="center">
  <em>Daily Load per Month (Year 1 & 2)</em>
</p>

<p align="center">
  <img src="report\Weekly_Load.png" width="800"/>
</p>

<p align="center">
  <em>Weekly Load per Month (Year 1 & 2)</em>
</p>

#### 2.4 Daily Temperature Pattern

- **Fixed Temperature Trough and Peak:**
  - **Temperature Trough (06:00–08:00):** The minimum temperature occurs in the early morning, shortly before or around sunrise (approximately **8–10°C** during the colder months).
  - **Temperature Peak (13:00–15:00):** The maximum temperature is reached during the afternoon, typically between **13:00 and 15:00**.

- **Phase Lag Relative to GHI (Thermal Lag Effect):**
  - Solar irradiance (GHI) reaches its peak around **12:00–13:00**, while the temperature peak is delayed by approximately **1–2 hours**, occurring around **13:00–15:00**.
  - This delay represents the time required for the Earth's surface to absorb solar radiation and subsequently transfer heat back to the surrounding air.

- **Moderate-Temperature Group (Site-1, Site-2, Site-4):**
  - These sites exhibit nearly **identical temperature variation patterns and ranges**.
  - During the summer months (August–September), peak temperatures in Year 1 reach an average of approximately **25.0–27.0°C**, while Year 2 remains around **23.5–24.5°C**.

- **High-Temperature Outlier – Hotspot (Site-3):**
  - Site-3 records the highest temperatures and the strongest day–night temperature variation among the five stations.
  - The summer peak temperature in Year 1 reaches approximately **34.0°C**, which is **7–9°C higher** than the other sites.
  - In Year 2, the peak remains the highest at approximately **31.5°C**.
  - *Possible explanation:* Site-3 may be located in an inland area, a valley, or a region experiencing stronger urbanization and heat-retaining effects from concrete and built-up surfaces. This interpretation should be treated as a hypothesis rather than a confirmed geographic characteristic.

- **Intermediate Site (Site-5):**
  - Temperature levels at Site-5 fall between the moderate-temperature group and Site-3.
  - The summer peak reaches approximately **30.0°C** in Year 1 and **26.5–27.0°C** in Year 2.

- **Cooler Summer Conditions in Year 2:**
  - Peak temperatures during the hottest months (July, August, and September) decrease noticeably by approximately **2.0–3.0°C** across all five sites in Year 2 compared with Year 1.
  - This is consistent with the observation from the load analysis that **peak electricity demand in Year 2 is slightly compressed compared with Year 1**, potentially due to less extreme weather conditions.

- **Stable Winter Temperature Baseline:**
  - The nighttime minimum temperatures during the colder months (December, January, and February) remain relatively stable between Year 1 and Year 2, at approximately **8.0–10.0°C**.

<p align="center">
  <img src="report\daily_temp_site1.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_temp_site2.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_temp_site3.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_temp_site4.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_temp_site5.png" width="800"/>
</p>

<p align="center">
  <em>Daily Temperature per Month (Year 1 & 2)</em>
</p>

#### 2.5 Weekly Temperature Pattern

- **Recurring 24-Hour Physical Cycle:**
  - Unlike electricity load, which varies according to work schedules, natural temperature follows a regular daily cycle from Monday to Sunday (`Mon`–`Sun`), with a trough in the early morning (**06:00–08:00**) and a peak in the early afternoon (**13:00–15:00**).
  - The same daily temperature pattern is therefore repeated throughout the **168-hour weekly window**.

- **Heatwave-Induced Noise:**
  - Some months exhibit unusually high temperature peaks on specific days of the week.
  - For example, in Year 1, the September peak on Sunday across Site-1, Site-2, Site-4, and Site-5 reaches approximately **27–29°C**.
  - This is likely the result of a random heatwave coinciding with that particular day rather than a systematic weekly behavioral pattern.

- **Moderate-Temperature Group (Site-1, Site-2, Site-4):**
  - The temperature profiles and variation ranges are nearly **identical** across these sites.
  - Summer temperature peaks in Year 1 reach approximately **25.0–27.0°C**, with an anomalous increase to **~28.5°C** in September.
  - In Year 2, the peaks decrease to approximately **23.5–25.5°C**.

- **High-Temperature Hotspot (Site-3):**
  - Site-3 consistently records the highest temperatures in the system, with an extremely wide day–night temperature range.
  - Summer temperature peaks reach **34.0–35.0°C** in Year 1 and **32.0–33.0°C** in Year 2.
  - During winter, nighttime temperatures can drop as low as **5.0–7.0°C**.

- **Intermediate Zone (Site-5):**
  - Temperature levels fall between the moderate-temperature group and Site-3.
  - Peak temperatures reach **30.0–31.5°C** in Year 1 and **26.5–28.0°C** in Year 2.

- **Year 2 Is Cooler Than Year 1 Across the Entire 168-Hour Profile:**
  - Peak temperatures across all summer/autumn months (June, July, August, and September) decrease consistently by approximately **2.0–3.0°C** in Year 2 compared with Year 1 across all five sites.

- **Stable Winter Temperature Baseline:**
  - Nighttime temperatures during the colder months (December, January, and February) follow a consistent pattern across both years, remaining within approximately **7.0–10.0°C**.

<p align="center">
  <img src="report\weekly_temp_site1.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_temp_site2.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_temp_site3.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_temp_site4.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_temp_site5.png" width="800"/>
</p>

<p align="center">
  <em>Weekly Temperature per Month (Year 1 & 2)</em>
</p>

#### 2.6 Daily GHI Pattern

- **Standard Symmetric Bell-Shaped Profile (Physical Boundary):**
  - GHI remains at **0 W/m²** from approximately **20:00 to 06:00** the following morning.
  - Solar irradiance begins to increase from around **06:00**, reaches its highest value (Peak GHI) around **12:30–13:30**, and drops back to zero after approximately **19:30**.
  - This produces a clear daily solar cycle, with irradiance concentrated around the daytime hours.

> *[Add figure: Daily GHI profiles across the five sites for Year 1 and Year 2]*

- **Distinct Seasonal Patterns:**
  - **Summer Group (May, June, July):**
    - These months exhibit the highest GHI peaks of the year, reaching approximately **950–980 W/m²**.
    - Daylight duration is also the longest, resulting in the widest time window with non-zero solar irradiance.
  - **Transition Group (March, April, August, September):**
    - Peak GHI ranges from approximately **700–880 W/m²**.
  - **Winter Group (November, December, January):**
    - December exhibits the lowest solar irradiance of the year, with peak GHI reaching only around **400–500 W/m²**, approximately half the summer level.
    - Sunrise occurs later, with GHI beginning to increase at around **07:30**, while sunset occurs earlier, with GHI returning to zero at around **18:00**.

- **Extremely High Spatial Similarity Across the Five Sites:**
  - The 24-hour profiles of all five sites (`Site-1` to `Site-5`) in both Year 1 and Year 2 are almost **identical** in terms of magnitude, peak timing, and monthly variation patterns.
  - **Implication:** The five measurement stations are likely located within a relatively small geographic area and are subject to similar regional weather conditions and cloud-cover patterns.

<p align="center">
  <img src="report\daily_ghi_site1.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_ghi_site2.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_ghi_site3.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_ghi_site4.png" width="800"/>
</p>

<p align="center">
  <img src="report\daily_ghi_site5.png" width="800"/>
</p>

<p align="center">
  <em>Daily GHI per Month (Year 1 & 2)</em>
</p>

#### 2.7 Weekly GHI Pattern

- **168-Hour Cyclic Pattern (Weekly Dynamics):**
  - **Independence from the Day of the Week:** Unlike electricity load, which is influenced by working schedules, solar irradiance (GHI) is a natural phenomenon. The magnitude and peak GHI across all seven days from Monday to Sunday (`Mon`–`Sun`) within the same month are nearly **identical**.
  - This indicates that there is no clear systematic weekly effect in the solar irradiance profile itself.

> *[Add figure: Weekly GHI profiles from Monday to Sunday]*

- **Monthly Variation in Peak Magnitude (Monthly Peak Shift):**
  - **Peak Season (May, June, July):**
    - Weekly average irradiance peaks remain consistently very high, reaching approximately **950–1,000 W/m²** across all days.
  - **Transition Season (March, April, August, September):**
    - Peak irradiance ranges from approximately **700–850 W/m²**.
  - **Low-Radiation Season (November, December, January):**
    - Peak irradiance decreases substantially, with December recording the lowest values at approximately **400–500 W/m²**, less than half the summer level.

- **Structural Consistency:**
  - The 168-hour profiles of all five sites (`Site-1 GHI` to `Site-5 GHI`) exhibit nearly identical patterns, including the timing of irradiance increase, sunset, and peak magnitude.

- **Geographical Interpretation:**
  - The five irradiance measurement stations are likely located within a relatively narrow climatic region and are subject to the same solar cycle and similar solar elevation angles.

- **Strong Temporal Consistency:**
  - The monthly variation patterns of solar irradiance in Year 1 and Year 2 are reproduced with high consistency, indicating a stable underlying seasonal cycle.

- **Weather/Data Anomalies in Year 2:**
  - In Year 2, on **Friday (`Fri`)**, the average irradiance profiles for several months—particularly the winter and transition months represented by the gray/purple curves—exhibit unusual drops or fluctuations around the **200 W/m²** level.
  - This pattern can be observed, for example, in the profiles of Site-1, Site-3, Site-4, and Site-5.
  - This may indicate **unusual weather conditions**, such as heavy cloud cover or prolonged rainfall on Fridays during those months, or potentially a **data artifact** caused by measurement noise or sensor-related issues.
  - Therefore, this pattern should be treated as an **observed anomaly requiring further investigation**, rather than being assumed to represent a systematic Friday effect.

<p align="center">
  <img src="report\weekly_ghi_site1.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_ghi_site2.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_ghi_site3.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_ghi_site4.png" width="800"/>
</p>

<p align="center">
  <img src="report\weekly_ghi_site5.png" width="800"/>
</p>

<p align="center">
  <em>Weekly GHI per Month (Year 1 & 2)</em>
</p>

#### 2.8 Multicollinearity and Spatial Redundancy

Correlation analysis reveals strong relationships among the weather measurements from different sites.

Temperature measurements are highly correlated with one another, and the same is true for GHI measurements. This is consistent with the highly similar temporal profiles observed during EDA.

A Variance Inflation Factor (VIF) analysis further confirms severe multicollinearity among the input variables. All evaluated variables have VIF values above 10, with values ranging from approximately **20.74 to 1059.49**.

The highest VIF values are observed for the GHI variables:

- `Site-5 GHI`: **VIF = 1059.49**
- `Site-4 GHI`: **VIF = 696.85**

Temperature variables also exhibit substantial multicollinearity, although generally lower than the GHI variables.

This indicates that directly using all five temperature and all five GHI measurements introduces a large amount of redundant information.

<p align="center">
  <img src="report\correlation_analysis.png" width="800"/>
</p>

<p align="center">
  <em>Correlation Analysis</em>
</p>

<p align="center">
  <img src="report\vif_analysis.png" width="800"/>
</p>

<p align="center">
  <em>VIF Analysis</em>
</p>

#### 2.9 Modeling Implications

The main EDA findings lead to the following feature engineering decisions:

| EDA Observation | Modeling Decision |
|---|---|
| Strong daily load periodicity | Daily Fourier features |
| Strong weekly load periodicity | Weekly Fourier features |
| Stable day-of-week behavior | `dow` and `is_weekend` |
| Strong correlation among temperature sites | PLS-based dimensionality reduction |
| Strong correlation among GHI sites | PLS-based dimensionality reduction |
| Low linear Load–Weather correlation | Nonlinear model such as XGBoost |
| Temperature–GHI phase difference | Weather lag features |
| Short-term weather variation | Weather delta features |
| Strong seasonal load pattern | Seasonal/calendar features |
| Year-over-Year weather variation | Chronological hold-out validation |

Overall, the EDA indicates that the forecasting problem is characterized by **strong temporal periodicity, highly correlated spatial weather measurements, nonlinear load–weather relationships, and relatively stable but non-identical yearly distributions**.

These observations motivate the feature engineering pipeline developed in the next section.

### 3. Feature Engineering

The feature engineering pipeline transforms the raw timestamp and weather measurements into a compact set of temporal, spatial, and weather-dynamics features.

The main objectives are:

- Reduce the dimensionality and multicollinearity of the five weather stations.
- Represent the strong daily and weekly periodicity observed in the EDA.
- Capture short-term changes in weather conditions.
- Provide calendar information related to weekly load patterns.
- Preserve a consistent time representation between training and testing data.

The complete feature engineering pipeline is implemented in [`features.py`](./utils/features.py).

#### 3.1 Calendar and Timestamp Construction

The competition dataset provides `Year`, `Month`, `Day`, and `Hour`, where `Hour` is represented using hour-ending labels from **1 to 24**.

The pipeline first maps the competition-relative year labels to actual calendar years using the following mapping:

```python
ANCHOR_YEARS = {
    1: 2020,
    2: 2021,
    3: 2022
}
````

Because `Hour = 1,...,24` represents hour-ending labels, the hour is shifted by one position when constructing the timestamp:

```python
hour = Hour - 1
```

This produces a timestamp at the **beginning of each hour**, ensuring that all 24 hourly observations belonging to the same calendar day remain associated with that day.

The resulting `Date` variable is then used to derive temporal features such as day of week and weekend indicators.


#### 3.2 PLS-Based Weather Dimensionality Reduction

The dataset contains temperature and GHI measurements from five different sites. As observed during EDA, measurements from different sites are highly correlated, creating substantial redundancy and multicollinearity.

Instead of directly using all five measurements for each weather variable, **Partial Least Squares (PLS) Regression** is used to construct lower-dimensional weather representations.

PLS is fitted using the available training data:

```python
pls_temp = PLSRegression(n_components=1)
pls_ghi = PLSRegression(n_components=2)
```

##### Temperature

The five temperature measurements are compressed into a single latent component using PLS.

##### GHI

The five GHI measurements are compressed into two latent components using PLS.

Using two components allows the model to retain more information from the highly correlated GHI measurements while still substantially reducing dimensionality.

The resulting weather representation is therefore:

| Original Variables         | Engineered Features                |
| -------------------------- | ---------------------------------- |
| 5 Temperature measurements | `Combined_Temp`                    |
| 5 GHI measurements         | `Combined_GHI_1`, `Combined_GHI_2` |

The PLS transformation is fitted using the selected training years and then applied to the complete dataset, including the testing period.

> *[Add figure: PLS transformation from five weather stations to latent weather factors]*

#### 3.3 Continuous Time Representation

A continuous time variable is constructed from the `Date` column:

```python
t = (Date - t0).total_seconds() / 3600
```

where `t0` is the first timestamp in the combined dataset.

This represents the elapsed time in hours from a fixed temporal origin.

Using elapsed time rather than the dataframe row index makes the temporal representation independent of row numbering and preserves the actual time difference between observations.

#### 3.4 Fourier Time Features

The EDA showed strong periodic patterns in electricity load, particularly at the **daily** and **weekly** levels.

Fourier features are therefore introduced to represent these periodic patterns continuously.

##### Daily Periodicity

Two daily harmonics are used:

```text
sin_day_k1
cos_day_k1
sin_day_k2
cos_day_k2
```

The first harmonic captures the fundamental 24-hour cycle, while the second harmonic provides additional flexibility for the non-sinusoidal daily load pattern.

##### Weekly Periodicity

One weekly harmonic is used:

```text
sin_week_k1
cos_week_k1
```

This represents the complete 168-hour weekly cycle

These features allow the model to represent smooth periodic changes without treating each hour as an unrelated categorical value.

#### 3.5 Calendar Features

Two additional calendar features are extracted from the constructed timestamp:

```python
dow = Date.dt.dayofweek
is_weekend = (dow >= 5).astype(int)
```

where:

* `dow` represents the day of the week, from **0 (Monday) to 6 (Sunday)**.
* `is_weekend` indicates whether the observation belongs to Saturday or Sunday.

These features complement the continuous Fourier representation by allowing the model to distinguish systematic differences between weekdays and weekends.

This is particularly relevant because the EDA showed that electricity demand exhibits different patterns across the weekly cycle.

#### 3.6 Weather Lag and Delta Features

The daily GHI and temperature plots showed clear short-term temporal dynamics. To capture these changes, the pipeline creates lag and difference features for the PLS-derived weather factors.

For each of:

```text
Combined_Temp
Combined_GHI_1
Combined_GHI_2
```

the following features are generated:

```text
lag_1_*
delta_1_*
delta_24_*
```

##### One-Hour Weather Lag

```text
lag_1_temp
lag_1_ghi_1
lag_1_ghi_2
```

represent the corresponding weather factor one hour earlier.

##### One-Hour Change

For example:

```text
delta_1_temp
delta_1_ghi_1
delta_1_ghi_2
```

These features describe short-term weather transitions rather than only the absolute weather level.

##### 24-Hour Change

```text
delta_24_temp
delta_24_ghi_1
delta_24_ghi_2
```

These features capture how the current weather condition differs from approximately the same time on the previous day.

#### 3.7 Cooling and Heating Degree Features

The pipeline also computes **Cooling Degree Hours (CDH)** and **Heating Degree Hours (HDH)** using the mean temperature across the five original temperature sites.

* `CDH` increases when the average temperature exceeds $20^\circ C$.
* `HDH` increases when the average temperature falls below $20^\circ C$.

These features provide an explicit representation of temperature conditions that may be associated with cooling and heating demand.

#### 3.8 Final Feature Groups

The resulting feature set can be organized into the following groups:

| Feature Group       | Features                                               | Purpose                                                  |
| ------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| Calendar            | `Year`, `Month`, `Day`, `Hour`                         | Preserve raw temporal information                        |
| Weather Compression | `Combined_Temp`, `Combined_GHI_1`, `Combined_GHI_2`    | Reduce spatial redundancy and multicollinearity          |
| Daily Periodicity   | `sin_day_k1`, `cos_day_k1`, `sin_day_k2`, `cos_day_k2` | Capture 24-hour load patterns                            |
| Weekly Periodicity  | `sin_week_k1`, `cos_week_k1`                           | Capture 168-hour periodicity                             |
| Yearly Periodicity  | `sin_year_k3`, `cos_year_k3`                           | Represent longer-term seasonal variation                 |
| Calendar Pattern    | `dow`, `is_weekend`                                    | Capture weekday/weekend effects                          |
| Weather Dynamics    | `lag_1_*`, `delta_1_*`, `delta_24_*`                   | Capture short-term and day-over-day weather changes      |
| Temperature Demand  | `CDH`, `HDH`                                           | Represent cooling/heating-related temperature conditions |

### 4. Model

#### 4.1 XGBoost Regression

The forecasting model is based on **XGBoost (Extreme Gradient Boosting)**, a gradient-boosted decision tree algorithm designed for nonlinear regression problems.

XGBoost was selected because the EDA indicates that the relationship between electricity demand and weather variables is not purely linear. Although individual temperature and GHI variables show relatively weak linear correlations with `Load`, electricity demand is influenced by nonlinear interactions between weather conditions, time of day, seasonality, and calendar effects.

This makes a tree-based nonlinear model a suitable choice for combining the engineered features.

#### 4.2 Gradient Boosting Framework

XGBoost builds the prediction model sequentially by adding decision trees that correct the errors made by previous trees.

At each boosting iteration, the newly added tree is optimized to reduce the remaining prediction error while controlling model complexity through regularization.

For this regression task, the squared-error objective is used.

#### 4.3 Hyperparameter Optimization with Optuna

The XGBoost hyperparameters were optimized using **Optuna**, with **RMSE** as the optimization objective under time-series cross-validation.

The best configuration obtained from the optimization process is:

| Hyperparameter     |  Best Value |
| ------------------ | ----------: |
| `n_estimators`     |         630 |
| `learning_rate`    |     0.14128 |
| `max_depth`        |           3 |
| `min_child_weight` |           3 |
| `subsample`        |     0.68608 |
| `colsample_bytree` |     0.73485 |
| `gamma`            |     1.05239 |
| `reg_alpha`        | 4.88 × 10⁻⁶ |
| `reg_lambda`       |     1.89410 |

The optimization objective is to minimize the Root Mean Squared Error (RMSE).

#### 4.4 Model Training

The model selection process follows the chronological structure of the dataset.

During hyperparameter optimization, the training period is divided using **TimeSeriesSplit** rather than random cross-validation. This preserves the temporal ordering of the observations and prevents future observations from being used to predict earlier observations.

After the optimal hyperparameters are selected, the final XGBoost model is retrained using the available training years before generating predictions for the unseen testing period.

#### 4.5 Why XGBoost?

The choice of XGBoost is motivated by several observations from the EDA:

- **Nonlinear weather–load relationship:** Linear correlations between individual weather variables and `Load` are relatively weak, suggesting that a nonlinear model is needed.
- **Feature interactions:** Electricity demand may depend on combinations of temperature, solar irradiance, time of day, season, and day of week.
- **Periodic features:** Fourier features provide smooth temporal representations that can be naturally combined with tree-based features.
- **Mixed feature types:** XGBoost can jointly process continuous weather factors, temporal variables, calendar indicators, and engineered weather-dynamics features.
- **Computational efficiency:** Gradient-boosted trees provide a strong nonlinear baseline without requiring the considerably higher training complexity of sequence-based neural networks.

### 5. Validation Strategy

Because electricity load is a time-dependent forecasting problem, the validation procedure preserves the chronological order of the observations. **Random train-test splitting is avoided** because it could allow information from later time periods to influence the training process.

The validation strategy is divided into three stages.

#### 5.1 Time-Series Cross-Validation

**Year 1** is used for hyperparameter optimization with `TimeSeriesSplit`.

Instead of randomly shuffling observations, the training data is progressively expanded over time:

```text
Fold 1: |---- Train ----|--- Validation ---|

Fold 2: |-------- Train --------|--- Validation ---|

Fold 3: |------------ Train ------------|--- Validation ---|

...
```

This setup ensures that each validation period occurs **after** its corresponding training period, making the evaluation more representative of real forecasting conditions.

The average RMSE across the validation folds is used as the objective for Optuna.

#### 5.2 Year 2 Hold-Out Validation

After Optuna identifies the best hyperparameter configuration using Year 1, the selected model is evaluated on **Year 2**, which is kept as a separate hold-out period.

Year 2 is not used during the Optuna search. This provides an additional evaluation of how well the selected model generalizes to a future year.

The main evaluation metrics are:

* **RMSE** — measures the magnitude of prediction errors while assigning greater weight to large errors.
* **MSE** — squared prediction error.
* **MAE** — average absolute prediction error.
* **MAPE** — percentage-based prediction error.
* **$R^2$** — proportion of variance explained by the model.

#### 5.3 Final Training and Forecasting

After model selection and validation, the final XGBoost model is retrained using the available labeled data from **Year 1 and Year 2** with the selected hyperparameters.

The trained model is then applied to **Year 3**, where the actual `Load` values are unavailable.

The complete chronological workflow is:

```text
Year 1
   │
   ├── TimeSeriesSplit
   │        │
   │        └── Optuna Hyperparameter Optimization
   │
   ▼
Best Hyperparameters
   │
   ▼
Year 2 Hold-Out Validation
   │
   ▼
Final Model
   │
   ├── Train: Year 1 + Year 2
   │
   └── Predict: Year 3
```

This validation design separates **hyperparameter optimization**, **model selection**, and **final forecasting**, while maintaining the temporal structure of the forecasting problem.

### 6. Experiments & Ablation

Ablation experiments were conducted incrementally to evaluate the contribution of different feature groups to hourly load forecasting.

In each experiment, the model was trained on **Year 1** and evaluated on the unseen **Year 2** period using five random seeds:

$$
Seeds = \{0, 1, 2, 3, 42\}
$$

The experiments progressively introduced:

1. Raw weather features as the baseline.
2. Time and periodicity features.
3. Weather lag and delta features.

The same XGBoost modeling procedure and evaluation metrics were used across all experiments.

#### 6.1 Baseline: Raw Weather Features

The baseline model uses the original weather measurements together with the calendar variables:

```text
Year, Month, Day, Hour
Site-1 Temp ... Site-5 Temp
Site-1 GHI  ... Site-5 GHI
```

The baseline does not apply dimensionality reduction, Fourier encoding, or weather temporal features.

| Metric |      Mean |    Std |       Min |       Max |
| ------ | --------: | -----: | --------: | --------: |
| RMSE   |    192.43 |   0.00 |    192.43 |    192.43 |
| MSE    | 37,029.25 |   0.00 | 37,029.25 | 37,029.25 |
| MAE    |    141.23 |   0.00 |    141.23 |    141.23 |
| MAPE   |     6.81% |  0.00% |     6.81% |     6.81% |
| $R^2$  |    0.7759 | 0.0000 |    0.7759 |    0.7759 |

This baseline provides a reference point for measuring the contribution of the subsequent feature engineering stages.

#### 6.2 Adding Time and Periodicity Features

The second experiment replaces the raw weather representation with PLS-based weather components and introduces explicit temporal representations.

The feature set includes:

* `Combined_Temp`
* `Combined_GHI_1`
* `Combined_GHI_2`
* Daily Fourier features
* Weekly Fourier features
* Day-of-week (`dow`)

The daily Fourier features model the intraday periodicity, while the weekly Fourier features represent the 168-hour cycle.

The results are:

| Metric |      Mean |    Std |       Min |       Max |
| ------ | --------: | -----: | --------: | --------: |
| RMSE   |    160.27 |   0.74 |    159.37 |    161.41 |
| MSE    | 25,686.12 | 236.20 | 25,398.08 | 26,052.80 |
| MAE    |    115.22 |   0.38 |    114.60 |    115.53 |
| MAPE   |     5.42% |  0.02% |     5.38% |     5.43% |
| $R^2$  |    0.8445 | 0.0014 |    0.8423 |    0.8463 |

Compared with the raw-feature baseline, RMSE decreases from **192.43 MW to 160.27 MW**, corresponding to an improvement of approximately **16.7%**.

This indicates that explicitly representing temporal periodicity and reducing the highly correlated weather variables into latent components provides substantially more useful information for the XGBoost model.

#### 6.3 Adding Weather Lag and Delta Features

The third experiment extends the time-feature configuration with short-term weather dynamics.

For the PLS-derived temperature and GHI components, the following features are added:

* 1-hour lag:

  * `lag_1_temp`
  * `lag_1_ghi_1`
  * `lag_1_ghi_2`
* 1-hour change:

  * `delta_1_temp`
  * `delta_1_ghi_1`
  * `delta_1_ghi_2`
* 24-hour change:

  * `delta_24_temp`
  * `delta_24_ghi_1`
  * `delta_24_ghi_2`

The results are:

| Metric |      Mean |    Std |       Min |       Max |
| ------ | --------: | -----: | --------: | --------: |
| RMSE   |    154.10 |   1.18 |    152.57 |    155.35 |
| MSE    | 23,748.69 | 363.75 | 23,277.13 | 24,132.55 |
| MAE    |    113.47 |   0.84 |    112.39 |    114.37 |
| MAPE   |     5.35% |  0.04% |     5.31% |     5.39% |
| $R^2$  |    0.8563 | 0.0022 |    0.8539 |    0.8591 |

Adding weather lag and delta features further reduces RMSE from **160.27 MW to 154.10 MW**, an additional improvement of approximately **3.8%**.

Compared with the original baseline, the complete feature configuration reduces RMSE from **192.43 MW to 154.10 MW**, corresponding to an overall reduction of approximately **19.9%**.

#### 6.4 Ablation Summary

The progressive effect of feature engineering is summarized below:

| Feature Configuration          |       RMSE |        MAE |      MAPE |      $R^2$ |
| ------------------------------ | ---------: | ---------: | --------: | ---------: |
| Raw weather baseline           |     192.43 |     141.23 |     6.81% |     0.7759 |
| + Time & periodicity features  |     160.27 |     115.22 |     5.42% |     0.8445 |
| + Weather lag & delta features | **154.10** | **113.47** | **5.35%** | **0.8563** |

The results show that the largest performance improvement comes from introducing **time-aware representations and PLS-based weather components**. Adding short-term weather dynamics through lag and delta features provides a further improvement.

Therefore, the final feature configuration consists of:

```text
Calendar Features
        +
PLS Weather Components
        +
Daily / Weekly Fourier Features
        +
Weather Lag & Delta Features
```

This configuration is used as the feature set for the subsequent experiments and final model.

#### 6.5 Final Prediction on Year 3

<p align="center">
  <img src="report\Final_prediction.png" width="800"/>
</p>

<p align="center">
  <em>Final Prediction</em>
</p>

## Tech Stack

* **Programming Language:** Python
* **Environment & Package Management:** Conda
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Hyperparameter Optimization:** Optuna
* **Statistical / Dimensionality Reduction:** PLS Regression
* **Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook
* **Version Control:** Git

## Quick Start

### Installation

Clone the repository and create a local Conda environment:

```bash
git clone https://github.com/ntq05/pge-hourly-load-forecasting.git
cd pge-hourly-load-forecasting

conda create --prefix ./env python=3.11
conda activate ./env

pip install -r requirements.txt
```

### Training

The training pipeline is implemented through Jupyter notebooks.

After installing the dependencies, open the notebooks and run them in the intended order to reproduce the experiments and train the models.

```bash
jupyter notebook
```

### Evaluation

The evaluation experiments are also provided as Jupyter notebooks.

Run the corresponding notebooks to evaluate the trained models on the **Year 2 hold-out period** and reproduce the reported metrics.

### Prediction

After selecting the final model and feature configuration, run the corresponding prediction notebook to generate hourly load forecasts for **Year 3**.

The prediction notebook uses the trained model and the available Year 3 time and weather features to generate the final forecasts.

## Project Structure

```text
.
├── README.md
├── requirements.txt
│
├── config/
│   └── config.py
│
├── Dataset/
│   ├── training.xlsx
│   └── testing.xlsx
│
├── Notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── baseline.ipynb
│   ├── 01_PLS_dimesion_reduction_training.ipynb
│   ├── 02_daily_first_harmonic_training.ipynb
│   ├── 03_weekly_first_harmonic_training.ipynb
│   ├── 04_daily_second_harmonic_training.ipynb
│   ├── 05_dayofweek_training.ipynb
│   ├── 06_lag1_temp_ghi_training.ipynb
│   ├── 07_delta1_temp_ghi_training.ipynb
│   ├── 08_delta24_temp_ghi_training.ipynb
│   ├── 09_CDH_training.ipynb
│   ├── 10_HDH_training.ipynb
│   └── report.ipynb
│
├── report/
│   ├── *.csv
│   └── *.png
│
└── utils/
    └── features.py
```

### Directory Description

| Directory / File   | Description                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `Dataset/`         | Raw training and testing datasets provided by the competition                                            |
| `Notebooks/`       | Exploratory analysis, feature engineering, model training, ablation experiments, and reporting notebooks |
| `report/`          | Generated prediction files, validation results, statistical reports, and visualization outputs           |
| `config/`          | Project configuration and shared settings                                                                |
| `utils/`           | Reusable feature engineering functions                                                                   |
| `requirements.txt` | Python package dependencies                                                                              |

## Dataset & References

### Dataset

This project uses the dataset from the **2025 PG&E Energy Analytics Challenge: Electric Load Forecasting**, provided through Zenodo.

The dataset contains hourly electricity load and exogenous weather information from five neighboring sites in the San Diego area:

* **Train.xlsx** — Three years of hourly electricity load data (2020–2022) with weather information.
* **Test.xlsx** — One year of hourly weather information (2023), with the load values reserved for forecasting during the challenge.

The dataset is publicly available at:

**Zenodo:** [2025 PG&E Energy Analytics Challenge Dataset](https://zenodo.org/records/17085273)

### References

* Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
* Wold, H. (1985). *Partial Least Squares*. In Encyclopedia of Statistical Sciences.
* Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
