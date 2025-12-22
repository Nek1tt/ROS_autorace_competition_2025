from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autorace_core_ROSchupepiki'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ('share/ament_index/resource_index/packages',
        #     ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nurshat',
    maintainer_email='n.abdyldaev@g.nsu.ru',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = autorace_core_ROSchupepiki.lane_follower:main',
            'monitor = autorace_core_ROSchupepiki.monitor:main',
            'konus_detection = autorace_core_ROSchupepiki.konus_detection:main',
            'tunnel_detection = autorace_core_ROSchupepiki.tunnel_detection:main',
            'aruco_detection = autorace_core_ROSchupepiki.aruco_detection:main',
        ],
    },
)
