from setuptools import setup, find_packages

setup(
    name="dynamic_deephit",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        'dynamic_deephit': ['data/*.csv'],
    },
    include_package_data=True,
    install_requires=[
        "numpy==1.16.5",
        "pandas==1.0.1",
        "tensorflow==1.13.1",
        "scikit-learn==0.22.1",
        "lifelines==0.24.9",
        "termcolor==1.1.0"
    ],
    python_requires=">=3.6.0,<3.7.0",
    maintainer="Zheng Chen",
    maintainer_email="zheng-chen@hotmail.com",
    classifiers=[
        "Author :: Chengfeng Zhang <2714311212@qq.com> (aut)",
        "Maintainer :: Zheng Chen <zheng-chen@hotmail.com> (cre)",
        "Contributor :: Yawen Hou (ctb)",
    ],
project_urls={
        "Paper-TBME-2020": "https://ieeexplore.ieee.org/document/8681104",
        "Citation": """
        @ARTICLE{8681104,
          author={Lee, Changhee and Yoon, Jinsung and van der Schaar, Mihaela},
          journal={IEEE Transactions on Biomedical Engineering}, 
          title={Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data}, 
          year={2020},
          doi={10.1109/TBME.2019.2909027}
        }"""
    }
)