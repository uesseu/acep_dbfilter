from setuptools import setup, find_packages

setup(
    name='dbfilter',
    version='0.0.1',
    package_dir={'dbfilter': 'dbfilter'},
    packages=find_packages(),
    description='Tools to manage ACEP project.',
    long_description='''Tools to manage ACEP project.
This is useful only for lower core members of ACEP.''',
    url='https://github.com/uesseu/aecp_dbfilter',
    author='Shoichiro Nakanishi',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "mergepsycho=dbfilter.mergepsycho:main",
            "mergedrug=dbfilter.mergedrug:main",
            "shrinkdrug=dbfilter.shrinkdrug:main",
        ]
    },
)
