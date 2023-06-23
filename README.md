# JRKit
![IMG-9570 (1)](https://github.com/jaberjaber23/JRKit_Advanced_Encoding_Toolkit/assets/103749727/8fcbe7cf-c061-4eb7-97a9-8aaedf39b71a)

JRKit is a powerful Python library for advanced data encoding techniques. It provides a collection of innovative encoding methods meticulously crafted to enhance data representation and feature extraction. These techniques address various encoding challenges, resulting in improved model performance and insightful data analysis.

Installation
You can install JRKit using pip:

```shell
pip install jrkit
```


## Usage
```shell
from jrkit import RegularizedLinearRegressionEncoder

X = ..
y = ..

# Fit the encoder to the data
encoder = RegularizedLinearRegressionEncoder()
encoder.fit(X, y)

# Transform the data
X_transformed = encoder.transform(X)
```

## Encoding Methods
* Leave-One-Out Encoder
This encoder leverages the "leave-one-out" strategy to encode categorical variables. It calculates statistics based on the remaining samples, minimizing data leakage risk and providing reliable representations for categorical features.

* EMWEncoder (Exponential Moving Window Encoder)
EMWEncoder utilizes an exponential moving window to encode sequential data. It captures temporal dynamics by assigning different weights to historical observations, enabling efficient feature extraction from time-series data.

* RBFEncoder (Radial Basis Function Encoder)
RBFEncoder employs radial basis functions to convert categorical variables into continuous representations. It utilizes Gaussian kernels to encode categories, preserving underlying relationships and enabling smooth transitions between different values.

* Regularized Linear Regression Encoder
The Regularized Linear Regression Encoder combines linear regression and regularization techniques to encode categorical features. By training a regularized linear regression model, it learns optimal coefficients for encoding categories while controlling model complexity.

JR's Advanced Encoding Toolkit offers a versatile set of encoding techniques for developers and data scientists. Each encoding method is accompanied by detailed documentation and code examples, ensuring seamless integration into existing projects.

## Documentation
For detailed usage instructions, API reference, and code examples, please refer to the JRKit Documentation.

## Contributing
We welcome contributions from the community. To contribute to JRKit, please read the Contribution Guidelines.

## License
JRKit is distributed under the MIT License.
