





本文完成一个点云变换的插件，同时也是对CC接口的使用做进一步说明，进一步理解CC插件开发流程，利用CC平台和已有代码加快实现定制功能。



> 这个功能在 cc 已有的功能已经存在，位于 edit->apply_transformation 这里



文件逻辑组织还和 [点云处理] cloud compare二次插件功能开发（一）结构搭建 中一致，不多赘述

> 项目代码全部放在 [Github](https://github.com/caibucai22/open-caima-coding)，并同步到 [Gitee]([caibucai22/open-caima-coding - 码云 - 开源中国 (gitee.com)](https://gitee.com/caibucai22/open-caima-coding)), 欢迎Star



# ui 层面

拿到cc源码qCC/ui_templates 下面的 applyTransformationDlg.ui 和对应的qcc下面的 .h .cpp

![image-20240823145623247](https://oss.caibucai.top/md/image-20240823145623247.png)



进行修改，我们只保留第一个使用Matrix 变换的tab，其他删除

![image-20240823145137206](https://oss.caibucai.top/md/image-20240823145137206.png)

> 也就只保留一个 Matrix 4x4 tab

对应ui的 .h 文件 和 .cpp 进行处理 

因为我们删去了3个tab 涉及到的 qt 组件都没了，需要对涉及到的函数进行 删除

删除基于 3个tab 使用的 组件名进行删，最终保留的函数如下

![image-20240823145727091](https://oss.caibucai.top/md/image-20240823145727091.png)



# ui逻辑总结

cc插件用到的ui，设计.ui文件，完成 .h .cpp，ui 逻辑函数，在需要用到的时候使用 xxdlg.exec() ,获取用户的各种输入，传回到 插件主逻辑中；cc采用了继承 ui文件编译后ui_xxxDlg.h 中的 Ui::xxxDialog，来封装一层，这种继承方式，使得声明的xxxDlg可以拿到界面上的所有组件，进而可以获取用户设置的组件值（或者使用函数返回）

## ui 界面层

```cpp
QT_BEGIN_NAMESPACE

class Ui_ApplyTransformationDialog // 各种组件
{
	// ...
};

namespace Ui {
    class ApplyTransformationDialog: public Ui_ApplyTransformationDialog {};
} // namespace Ui

```

## ui 逻辑层

```cpp
class ccApplyTransformationDlg : public QDialog, public Ui::ApplyTransformationDialog
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ccApplyTransformationDlg(QWidget* parent = nullptr);

	//! Returns input matrix
	ccGLMatrixd getTransformation(bool& applyToGlobal) const;

protected:

	void checkMatrixValidityAndAccept();

	void onMatrixTextChange();

	void onRotAngleValueChanged(double);

	void onEulerValueChanged(double);

	void onFromToValueChanged(double);

	void loadFromASCIIFile();
    
	void loadFromClipboard();

	void buttonClicked(QAbstractButton*);


protected:

	void updateAll(const ccGLMatrix& mat, bool textForm = true, bool axisAngleForm = true, bool eulerForm = true, bool fromToForm = true);
};
```

## ui 调用

插件主逻辑中调用，并获取用户的输入值（通过函数方式 ccApplyTransformationDlg.getTransformation() ）

```cpp
ccApplyTransformationDlg ccApplyTransformationDlg(m_app->getMainWindow());
if (!ccApplyTransformationDlg.exec())
{
    return;
}
bool applyToGlobal = false;
ccGLMatrixd transformMatrix =  ccApplyTransformationDlg.getTransformation(applyToGlobal);
```

用图说话

<img src="https://oss.caibucai.top/md/image-20240823154604852.png" alt="image-20240823154604852" style="zoom: 67%;" />





# 接口关系说明

mainwindow.h 继承了 ccMainAppInterface ，这也说明了 cc 就是使用的插件化开发范式

![image-20240822210601900](https://oss.caibucai.top/md/image-20240822210601900.png)





插件继承 ccStdPluginInterface

![image-20240822210744724](https://oss.caibucai.top/md/image-20240822210744724.png)



ccStdPluginInterface中 有 ccMainAppInterface的指针 m_app，可以获取到比 ccStdPluginInterface 中更高级的函数

![image-20240822210904578](https://oss.caibucai.top/md/image-20240822210904578.png)

所以 m_app 可以拿到很多有用的方法 

1. 将获取到的点云 从 db_tree 分离 / 放回
2. zoomOnSelectedEntities()
3. refreshAll()
4. ...

但是mainwindow 自定义的方法 确无法直接拿到【但其中也是使用很多基于接口的方法来完成】，需要做本地化修改，对直接使用接口的方法 替换成 用 m_app 调用

所以，理论上 cc 能做的事，我们如果插件逻辑需要用到 都可以拿过来 进行改造后 使用，减少开发，本文便是一个简单示例



# 插件主逻辑

接下来的插件主逻辑 编写 和 

[cloud compare PCA插件开发详细步骤（二）附代码-CSDN博客](https://blog.csdn.net/csy1021/article/details/141284020?spm=1001.2014.3001.5502) 中 qPCA.cpp 实现部分，实现基本一致，完成

1. 构造函数
2. onNewSelection()
3. getActions()
4. doAction()

的实现

## 搬运CC实现并修改

我们接下来要继续的就是 doAction() 中的核心函数 

```cpp
void applyTransformation(ccMainAppInterface* m_app, const ccGLMatrixd& mat, bool applyToGlobal);
```

的实现过程



首先我们知道 cc 是已经实现过这个函数的了，在 mainwindow.cpp 中 ，因此先直接搬过来

```cpp
void MainWindow::applyTransformation(const ccGLMatrixd& mat, bool applyToGlobal)
{
    
}
```

会发现 很多 报错地方，进行一一解决

```cpp
void applyTransformation(ccMainAppInterface* m_app, const ccGLMatrixd& mat, bool applyToGlobal)
{
    
}
```

我们首先删除MainWindow::，增加 插件中使用的接口指针m_app

首先会发现很多 直接调用的函数，报错

包括

```cpp
getTopLevelSelectedEntities();

haveOneSelection();

putObjectBackIntoDBTree()

zoomOnSelectedEntities();

refreshAll();
```

我们尝试用 m_app 去调用

```cpp
m_app->putObjectBackIntoDBTree(entity, objContext);
m_app->zoomOnSelectedEntities();
m_app->refreshAll();
```

可以调用

然后再去看其他函数，发现他们虽不能用 m_app 调用，但他们的实现中也直接或间接用到了 ccMainAppInterface 的方法，因此我们传入 m_app ，进行本地化 替换修改

比如，getTopLevelSelectedEntities(); 修改后

```cpp
ccHObject::Container getTopLevelSelectedEntities(ccMainAppInterface* m_app)
{
	 // m_selectedEntities 在 mainwindow 是成员变量可以直接获取到 这里我们使用 m_app 进行获取
    const ccHObject::Container& m_selectedEntities = m_app->getSelectedEntities();
	ccHObject::Container topLevelSelectedEntities;
	for (size_t i = 0; i < m_selectedEntities.size(); ++i)
	{
		ccHObject* entity = m_selectedEntities[i];
		bool hasParentsInselection = false;
		for (size_t j = 0; j < m_selectedEntities.size(); ++j)
		{
			if (i == j)
				continue;

			ccHObject* otherEntity = m_selectedEntities[j];
			if (otherEntity->isAncestorOf(entity))
			{
				hasParentsInselection = true;
				break;
			}
		}

		if (!hasParentsInselection)
		{
			topLevelSelectedEntities.push_back(entity);
		}
	}

	return topLevelSelectedEntities;
}
```

其他的一些报错，无关乎主要逻辑的删除即可

一些数据结构报错的，找到对应的头文件引入即可

这样我们的核心函数就修改完成了，往往比自己写的更优秀



在修改的过程 也可以进一步学习cc的逻辑，优化我们之前的写插件的处理思路

> 如，cc在处理点云时，不直接对 db_tree上点云进行处理，而是从 db_tree 分离后再处理，然后再放回





我们的网站是[菜码编程](https://www.caima.tech/)，我们的q群是 661282238
如果你对我们的项目感兴趣，欢迎扫码关注公众号，加入群聊，与我们讨论，我们会为你带来更多优秀的实践项目。



![fotter](https://oss.caibucai.top/md/fotter.png)

