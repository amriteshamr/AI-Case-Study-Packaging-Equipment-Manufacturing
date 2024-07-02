**Discussion**

The objective of this study was to determine the most effective machine
learning model for predicting Overall Equipment Effectiveness (OEE) and
Machine Maintenance Efficiency (MME) in a packaging equipment
manufacturing industry. Using synthetic data generated with realistic
parameter ranges, we evaluated the performance of ARIMA, GRU, and TCN
models based on RMSE, MAE, and MAPE metrics.

One of the initial challenges we faced was the unavailability of real
data from actual packaging equipment manufacturing companies. To address
this, we generated synthetic data, ensuring that it closely mirrored
real-world conditions by researching necessary parameters and their
realistic value ranges. However, this approach introduces a limitation:
synthetic data might not fully capture the complexities and nuances of
actual operational data, such as interdependencies and temporal dynamics
like seasonality or periodic maintenance cycles.

Our findings indicate that ARIMA, a traditional time series model,
performed the worst. This result might be attributed to the synthetic
nature of the data, as ARIMA often struggles with the complexity and
non-linearity that could be inadequately represented in synthetic
datasets. Conversely, both GRU and TCN models showed strong performance,
with TCN slightly outperforming GRU. While these results are promising,
they also raise concerns about whether such high accuracy and low error
rates would be achievable with real-world data. The excellent
performance of GRU and TCN on synthetic data might not fully translate
to actual industrial settings where data is more complex and less
predictable.

Future research should focus on validating our models using real-world
data from manufacturing industries to understand their practical
applicability better. Additionally, incorporating more parameters that
influence OEE and MME, such as maintenance schedules, operator
efficiency, and environmental factors, could further enhance model
accuracy. Exploring hybrid models that combine different machine
learning approaches might also capture both linear and non-linear
patterns more effectively.

**Conclusion**

In conclusion, our study aimed to identify the best machine learning
model for predicting OEE and MME in a manufacturing setting. We found
that ARIMA was the least effective model, potentially due to the
limitations of synthetic data. Both GRU and TCN models showed
significant promise, with TCN slightly outperforming GRU. Despite the
robust synthetic data generation and comprehensive evaluation metrics
used in our study, the lack of real-world data poses a significant
limitation. The synthetic data might not fully capture the
interdependencies and temporal dynamics of actual processes, and the
high performance of GRU and TCN on synthetic data may not be replicable
with real-world data.

Future research should prioritize testing these models with real-world
data and exploring additional parameters and hybrid models to enhance
predictive accuracy. Incorporating real-world data will help validate
the models' effectiveness and ensure their practical applicability in
industrial settings. Additionally, expanding the scope of the models to
include more comprehensive factors influencing OEE and MME could provide
more accurate and reliable predictions. Ultimately, our findings offer a
foundational understanding and direction for improving predictive
maintenance and operational efficiency in the manufacturing industry,
paving the way for future advancements in machine learning applications
in this field.
