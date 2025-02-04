from setuptools import find_packages, setup

package_name = 'ur5_arm_zell_machine_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zezheng Fu(Zell)',
    maintainer_email='11911719@mail.sustech.edu.cn',
    description='A ur5 machine learning package',
    license='MIT',
    # tests_require=['pytest'],
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
