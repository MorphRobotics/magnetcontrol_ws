from setuptools import find_packages, setup
import os
package_name = 'magnet_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    # --- ADD THESE LINES ---
    (os.path.join('lib', package_name), ['magnet_control/mscr_inverse_model.onnx']),
    (os.path.join('lib', package_name), ['magnet_control/inv_norm3.mat']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dozie',
    maintainer_email='dozie@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'move_waypoints = magnet_control.move_waypoints:main',
        'mscr_inv_control = magnet_control.mscr_inv_control:main',
        'mscr_inv_cosserat = magnet_control.mscr_inv_cosserat:main',
        'mscr_raw = magnet_control.mscr_raw:main',
        'lung_navigation_sim = magnet_control.lung_navigation_sim:main',
        ],
    },
)
