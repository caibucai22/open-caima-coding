点云处理，有一个出名的处理软件，cloud compare，简称cc，将自己实现的功能以插件形式集成到CC里，方便使用



## 前提

环境：cc 2.13，qt 5.15，cmake 3.18，vs2019【其他组合也可，本文基于此展开】

能力要求：能够使用cmake成功编译 cc并安装



# CC插件概述

CC提供了一种插件化的二次开发方式，以插件的形式，避免了核心代码的修改，利用提供的接口，完成我们需要功能的二次开发

## 组织结构

![image-20240814165218429](https://oss.caibucai.top/md/image-20240814165218429.png)



cc中的插件全都放在 源码plugins文件夹下

core 是 cc已经实现的插件功能

example 是 cc 提供给我们的示例，本文也基于其中的示例进行开发我们的插件



其中 core 文件夹下，又对插件进行了划分

- GL，基于gl可视化插件
- IO ，涉及IO处理的插件
- Standard ，大部分插件都属于这种



example 文件夹下也对应提供了 相应的示例插件

- ExampleGLPlugin
- ExampleIOPlugin
- ExamplePlugin



> 本文基于其中的ExamplePlugin插件，也就是标准插件类型，实现一个PCA功能，并可视化



## ExamplePlugin



![image-20240814170233699](https://oss.caibucai.top/md/image-20240814170233699.png)

images 中放置的是 icon.png

include 涉及的头文件，自己的功能头文件和用到的其他第三方的头文件

src 功能代码

ExamplePlugin.qrc，qt 组织资源的方式，提供资源的路径给代码使用

info. json，插件的描述，涉及的相关资源路径，以及开发者信息、相关资料

```json
{
	"type": "Standard",
	"name": "MyTest (Standard Plugin)",
	"icon": ":/CC/plugin/MyTestPlugin/images/icon.png",
	"description": "This is a description of the marvelous Example plugin. It does nothing.",
	"authors": [
		{
			"name": "xxx",
			"email": "xxx"
		}
	],
	"maintainers": [
		{
			"name": "yyy,
			"email": "yyy@gmail.com"
		},
		{
			"name": "zzz"
		}
	],
	"references": [
		{
			"text": "xx references",
			"url": "http://www.bmj.com/content/333/7582/1285"
		},
		{
			"text": "a test plugin"
		}
	]
}

```



## cmake 组织

基础的结构讲完，cc 是如何通过某种方式将插件组织起来的，答案就是 cmake



从高到低涉及的 cmakelists.txt 完成了这一任务【下面或区分两条不同路径情况，一个是放在 core\standard ，另一个是放在example】

D:\06-source_code\CloudCompare-2.13\CMakeLists.txt

```cmake
# ...

# Plugins
add_subdirectory( plugins )

# ...
```





D:\06-source_code\CloudCompare-2.13\plugins\CMakeLists.txt

```cmake
# ...

add_subdirectory( core )
add_subdirectory( example )

# ...
```



D:\06-source_code\CloudCompare-2.13\plugins\core\CMakeLists.txt 【D:\06-source_code\CloudCompare-2.13\plugins\example\CMakeLists.txt】

```cmake
add_subdirectory( GL )
add_subdirectory( IO )
add_subdirectory( Standard )

```

或

```cmake
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/ExampleGLPlugin )
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/ExampleIOPlugin )
add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/ExamplePlugin )
```





D:\06-source_code\CloudCompare-2.13\plugins\core\Standard\CMakeLists.txt 

```cmake
add_subdirectory( qAnimation )
add_subdirectory( qBroom )
add_subdirectory( qCanupo )
add_subdirectory( qCloudLayers )
add_subdirectory( qCompass )
add_subdirectory( qCork )
add_subdirectory( qCSF )
add_subdirectory( qFacets )
add_subdirectory( qHoughNormals )
add_subdirectory( qHPR )
add_subdirectory( qM3C2 )
add_subdirectory( qPCL )
add_subdirectory( qPCV )
add_subdirectory( qPoissonRecon )
add_subdirectory( qRANSAC_SD )
add_subdirectory( qSRA )
add_subdirectory( qMeshBoolean )

#plugins integrated as submodules
set( submod_plugins
		${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter
		${CMAKE_CURRENT_SOURCE_DIR}/qMasonry
		${CMAKE_CURRENT_SOURCE_DIR}/qMPlane
		${CMAKE_CURRENT_SOURCE_DIR}/qJSonRPCPlugin
		${CMAKE_CURRENT_SOURCE_DIR}/qTreeIso
		${CMAKE_CURRENT_SOURCE_DIR}/q3DMASC
)

foreach( dir ${submod_plugins} )
    if( IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt )
		message( STATUS "Found submodule plugin: " ${dir} )
		add_subdirectory( ${dir} )
	endif()
endforeach()
```



具体插件的CMakeLists.txt

【D:\06-source_code\CloudCompare-2.13\plugins\core\Standard\qPCA\CMakeLists.txt】或【D:\06-source_code\CloudCompare-2.13\plugins\example\ExamplePlugin\CMakeLists.txt】

```cmake
# ...
```



或

```cmake
# CloudCompare example for standard plugins

# REPLACE ALL 'ExamplePlugin' OCCURENCES BY YOUR PLUGIN NAME
# AND ADAPT THE CODE BELOW TO YOUR OWN NEEDS!

# Add an option to CMake to control whether we build this plugin or not
option( PLUGIN_EXAMPLE_STANDARD "Install example plugin" OFF )

if ( PLUGIN_EXAMPLE_STANDARD )
	project( ExamplePlugin )
	 
	AddPlugin( NAME ${PROJECT_NAME} )
		
	add_subdirectory( include )
	add_subdirectory( src )
	
	# set dependencies to necessary libraries
	# target_link_libraries( ${PROJECT_NAME} LIB1 )
endif()
```



# 具体开发前修改

## info.json

```cmake
{
	"type": "Standard",
	"name": "PCA (Standard Plugin)",
	"icon": ":/CC/plugin/qPCA/images/icon.png",
	"description": "This is a description of the PCA plugin. It does nothing.",
	"authors": [
		{
			"name": "xxx",
			"email": "xxx"
		}
	],
	"maintainers": [
		{
			"name": "yyy,
			"email": "yyy@gmail.com"
		},
		{
			"name": "zzz"
		}
	],
	"references": [
		{
			"text": "xx references",
			"url": "http://www.bmj.com/content/333/7582/1285"
		},
		{
			"text": "a PCA plugin"
		}
	]
}

```



## 具体插件上一级的CMakeLists.txt，这里是 Standard\CMakeLists.txt 

```cmake
add_subdirectory( qAnimation )
add_subdirectory( qBroom )
add_subdirectory( qCanupo )
add_subdirectory( qCloudLayers )
add_subdirectory( qCompass )
add_subdirectory( qCork )
add_subdirectory( qCSF )
add_subdirectory( qFacets )
add_subdirectory( qHoughNormals )
add_subdirectory( qHPR )
add_subdirectory( qM3C2 )
add_subdirectory( qPCL )
add_subdirectory( qPCV )
add_subdirectory( qPoissonRecon )
add_subdirectory( qRANSAC_SD )
add_subdirectory( qSRA )
add_subdirectory( qMeshBoolean )

# --------------------
add_subdirectory( qPCA )
# --------------------

#plugins integrated as submodules
set( submod_plugins
		${CMAKE_CURRENT_SOURCE_DIR}/qColorimetricSegmenter
		${CMAKE_CURRENT_SOURCE_DIR}/qMasonry
		${CMAKE_CURRENT_SOURCE_DIR}/qMPlane
		${CMAKE_CURRENT_SOURCE_DIR}/qJSonRPCPlugin
		${CMAKE_CURRENT_SOURCE_DIR}/qTreeIso
		${CMAKE_CURRENT_SOURCE_DIR}/q3DMASC
)

foreach( dir ${submod_plugins} )
    if( IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt )
		message( STATUS "Found submodule plugin: " ${dir} )
		add_subdirectory( ${dir} )
	endif()
endforeach()
```



## 具体插件的CMakeLists.txt， qPCA/CMakeLists.txt

```cmake
# CloudCompare example for standard plugins

# REPLACE ALL 'ExamplePlugin' OCCURENCES BY YOUR PLUGIN NAME
# AND ADAPT THE CODE BELOW TO YOUR OWN NEEDS!

# Add an option to CMake to control whether we build this plugin or not
# option( PLUGIN_EXAMPLE_STANDARD "Install example plugin" OFF )
option( PLUGIN_qPCA "Install example plugin" OFF )

# if ( PLUGIN_EXAMPLE_STANDARD )
if ( PLUGIN_qPCA )
	# project( ExamplePlugin )
	project( QPCA_PLUGIN ) # 全部大写
	 
	AddPlugin( NAME ${PROJECT_NAME} )
		
	add_subdirectory( include )
	add_subdirectory( src )
	
	# 添加其他需要的库 set dependencies to necessary libraries 
	# target_link_libraries( ${PROJECT_NAME} LIB1 )
endif()
```



## 更新ExamplePlugin 中的文件名为新插件对应的名称

ExamplePlugin.qrc ---> qPCA.qrc

ExamplePlugin.h ---> qPCA.h

ExamplePlugin.cpp ---> qPCA.cpp

src  include 下的 CMakeLists.txt 更新文件名

src/CMakeLists.txt

```cmake
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/ActionA.cpp
		# ${CMAKE_CURRENT_LIST_DIR}/ExamplePlugin.cpp
		${CMAKE_CURRENT_LIST_DIR}/qPCA.cpp
		# 其他新加文件
)
```



include/CMakeLists.txt

```cmake
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/ActionA.h
		# ${CMAKE_CURRENT_LIST_DIR}/ExamplePlugin.h
		${CMAKE_CURRENT_LIST_DIR}/qPCA.h
		# 候选其他头文件也要在这添加
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)
```



## 更新ExamplePlugin.qrc，更新资源路径

直接用 vscode 打开 修改

```xml
<RCC>
    <qresource prefix="/CC/plugin/ExamplePlugin">
        <file>images/icon.png</file>
        <file>info.json</file>
    </qresource>
</RCC>

```

修改后

![image-20240814190146241](https://oss.caibucai.top/md/image-20240814190146241.png)

此图片为后面生成project 后在vs中修改

![image-20240814190024700](https://oss.caibucai.top/md/image-20240814190024700.png)



## 主文件修改

修改qPCA.cpp

```c++
# #include "ExamplePlugin.h" 改为
#include "qPCA.h"

# ,ccStdPluginInterface( ":/CC/plugin/ExamplePlugin/info.json" )
    
,ccStdPluginInterface( ":/CC/plugin/qPCA/info.json" )
```



修改qPCA.h

```c++
# Q_PLUGIN_METADATA( IID "cccorp.cloudcompare.plugin.Example" FILE "../info.json" )
Q_PLUGIN_METADATA( IID "cccorp.cloudcompare.plugin.qPCA" FILE "../info.json" )
```



# cmake 构建

![image-20240814205307575](https://oss.caibucai.top/md/image-20240814205307575.png)

可以看到我们的 qPCA 插件选项，选中

然后生成project

# 运行

<img src="https://oss.caibucai.top/md/image-20240814202916848.png" alt="image-20240814202916848" style="zoom:50%;" />



![image-20240814203105623](https://oss.caibucai.top/md/image-20240814203105623.png)



下一篇，我们来实现具体的PCA逻辑和可视化效果



我们的网站是[菜码编程](https://www.caima.tech)。
如果你对我们的项目感兴趣可以扫码关注我们的公众号，我们会持续更新深度学习相关项目。
![公众号二维码](https://i-blog.csdnimg.cn/direct/ede922e7158646bcaf95ec7eefc67e49.jpeg#pic_center)
