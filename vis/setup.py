from setuptools import setup, find_packages  
 
setup(  
    name='vis_tools',  
    version='0.3',  # 初始版本，
    description='A simple example package',  
    include_package_data=True,
    #long_description=open('README.md').read(),  # 读取README文件作为长描述  
    long_description_content_type='text/markdown',  # 指示内容类型为markdown  
    #url='https://github.com/your_username/my_library',  # 你的库的URL（如果有的话）  
    author='AoruXue',  
    author_email='aoru45@t.shu.edu.cn',  
    license='MIT',  # 许可证类型  
    packages=find_packages(),  # 自动查找包  
    install_requires=[],  # 依赖项列表（如果有的话）  
    tests_require=['pytest'],  # 测试依赖项（如果有的话）  
    test_suite='nose.collector',  # 测试套件（可选）  
    zip_safe=False,  # 如果你的包可以安全地作为zip文件运行，则为True  
)
