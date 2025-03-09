from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kimhoyun',
    maintainer_email='suberkut76@gmail.com',
    description='A package that predicts future trajectories using regression analysis based on odometry data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "trajectory_estimation = trajectory_estimation.trajectory_estimation:main",
        ],
    },
)
