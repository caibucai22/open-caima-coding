visual studio

## 项目一览

以开源项目 cloudcompare 为例，一个大型项目 肯定会有很多模块，每个模块 根据需求 生成 dll 库 或者 lib 库

![image-20250308142049088](https://oss.caibucai.top/md/image-20250308142049088.png)



主程序

![image-20250307172029745](https://oss.caibucai.top/md/image-20250307172029745.png)



dll 库

![image-20250307172951873](https://oss.caibucai.top/md/image-20250307172951873.png)



lib库

![image-20250307173140957](https://oss.caibucai.top/md/image-20250307173140957.png)



## debug 模式调试

编译生成，运行



![image-20250307174057596](https://oss.caibucai.top/md/image-20250307174057596.png)

提示找不到 运行依赖的dll

查找相应项目库的路径

![image-20250307174336840](https://oss.caibucai.top/md/image-20250307174336840.png)

可以看到是已经编译生成了的

只不过 和我们的 exe 不在同文件夹下

此时 我们可以选择安装，或者调整输出目录



### 中间目录输出

先看一下xx.dir/[debug/release] 下有什么

![image-20250307174755839](https://oss.caibucai.top/md/image-20250307174755839.png)

日志 以及 .obj 文件



### 方案A 安装法

![image-20250307194304089](https://oss.caibucai.top/md/image-20250307194304089.png)

安装后我们的exe 以及需要的 dll  在同一文件夹下，如下

![image-20250307194629239](https://oss.caibucai.top/md/image-20250307194629239.png)



此时如果调试，会发现仍然启动不了，找不到dll，我们修改调试路径

![image-20250308153714390](https://oss.caibucai.top/md/image-20250308153714390.png)

然后就可以了

### 方案B 调整输出目录

【另外一种方案 修改各种 dll lib 输出到 和 exe 同级，

普通操作，在vs 里面 一个个修改模块 输出目录 和 中间目录

高级操作(来自deepseek 未测试)，在.vcxporj 中修改 deepseek

```xml
<!-- 在 .vcxproj 文件中添加 -->
<Target Name="PostBuild" AfterTargets="PostBuildEvent">
    <Exec Command="xcopy /Y /S "$(TargetPath)" "D:\02-ide\cc_StarEdge\CloudCompare_debug\"" />
</Target>
```



 

### 开始调试

然后 点击debug 调试

![image-20250307200743739](https://oss.caibucai.top/md/image-20250307200743739.png)

在主程序 和 dll 库 打断点 测试

主程序 debug

![image-20250307201229923](https://oss.caibucai.top/md/image-20250307201229923.png)



成功断住

![image-20250307201800055](https://oss.caibucai.top/md/image-20250307201800055.png)



下一行 是 CCAppCommon dll库里面的内容, 断点进入

![image-20250307202026171](https://oss.caibucai.top/md/image-20250307202026171.png)



![image-20250307202058826](https://oss.caibucai.top/md/image-20250307202058826.png)

可以看到 我们由 .exe 进入到 .dll中 此时 就可以调试dll 文件了

可以调试的原因 是 因为 在 debug 模式下 dll 也生成了 pdb 这个pdb可以被加载

【我们使用的dll 是安装目录下的 dll  pdb 是在build 目录下 被加载，要保持版本一致，dll 变化了 pdb 也要对应更新（自动），此时要把新的 dll 安装到 安装目录下 保持一致 否则 会提示 符号未加载】

![image-20250307174336840](https://oss.caibucai.top/md/image-20250307174336840.png)

### 针对 pdb 在 debug下自动生成 验证测试

为了验证 我们删除 pdb 文件 再次debug 看看是否能断住 【信我，我真的删了】

![image-20250307203746099](https://oss.caibucai.top/md/image-20250307203746099.png)



### 针对 符号未加载原因 测试验证

有时候可能是 DLL 和 .pdb 编译版本不同

测试验证

向 dll 代码 新添加了 一行 此时进行编译 但未进行安装

![image-20250307210359288](https://oss.caibucai.top/md/image-20250307210359288.png)



更新前 时间 20:35

![image-20250307210501184](https://oss.caibucai.top/md/image-20250307210501184.png)

更新后 21:05

![image-20250307210540899](https://oss.caibucai.top/md/image-20250307210540899.png)

![image-20250307210602741](https://oss.caibucai.top/md/image-20250307210602741.png)



未安装，此时启动 是否能断住

![image-20250307210707148](https://oss.caibucai.top/md/image-20250307210707148.png)

没有断住 直接启动成功



我们重新安装后，再测试 是否断住

![image-20250307210837266](https://oss.caibucai.top/md/image-20250307210837266.png)

![image-20250307210905312](https://oss.caibucai.top/md/image-20250307210905312.png)



结果：成功断住！！！

### 小结

使用这种 在 安装目录下调试 修改代码后 要重新安装 更新相关dll lib 保证调试加载的pdb 和实际使用的 dll lib 版本一致 

> 最佳实践 应该是 编译更新后 自动将 新dll 安装到 安装目录下 ，也是我们省略的另一种方案实现，不展示使用这种方案的原因之一是，有时候 比较多dll lib 需要逐一修改，没有尝试过自动化实现



以上操作都是 在debug 模式下 完成



## release 模式调试

### release vs debug 配置区别

为什么会出现这个需求，原因就是 加载更快，观察对比 release 和 debug 配置的区别

![cpp_debug](https://oss.caibucai.top/md/cpp_debug.png)

根据上面的区别 我们需要把选项一一打开

1. C/C++ --> 常规 调试信息格式设置为“程序[数据库(/Zi) 
2. C/C++ --> 优化 ，将右侧优化设置为 已禁用 
3. 选择链接器 --> 调试，将右侧生成调试信息 设置为 是



### 未启用 选项

首先 在不打开 尝试 运行调试

可以看到 依然提示找不到 依赖的dll ,此时还是和上面一样的方案

1. 安装
2. 调整输出目录

仍然以 安装 方案 展示

![image-20250308112855074](https://oss.caibucai.top/md/image-20250308112855074.png)

![image-20250308112945970](https://oss.caibucai.top/md/image-20250308112945970.png)

release 安装 【文件夹后缀没有_debug 以及 dll 没有 d 标志】



继续 设置调试路径在 release 安装目录下

![image-20250308113137672](https://oss.caibucai.top/md/image-20250308113137672.png)



### 开始调试

断点

![image-20250308113231145](https://oss.caibucai.top/md/image-20250308113231145.png)

尝试点击 运行

![image-20250308113323444](https://oss.caibucai.top/md/image-20250308113323444.png)

提示不会命中，程序直接启动  不会断住，



原因就是 在 release 没有 生成 pdb 文件

![cpp_debug_release_vs_debug_pdb_generate](https://oss.caibucai.top/md/cpp_debug_release_vs_debug_pdb_generate.png)

### 启用 选项

此时 我们 打开主程序项目 上面提到的开关

1. C/C++ --> 常规 调试信息格式设置为“程序[数据库(/Zi) 
2. C/C++ --> 优化 ，将右侧优化设置为 已禁用 
3. 选择链接器 --> 调试，将右侧生成调试信息 设置为 是

>  我真的打开了

打开后 点击运行

![image-20250308113854046](https://oss.caibucai.top/md/image-20250308113854046.png)

可以看到 vs 在重新生成 pdb 文件



此时 程序依然直接 运行 没有断住 

![image-20250308114341400](https://oss.caibucai.top/md/image-20250308114341400.png)

![image-20250308114152361](https://oss.caibucai.top/md/image-20250308114152361.png)

已经生成了 pdb 文件，why

原因就是我们的 安装目录 dll 和 新生成 pdb 不一致，也就是 你忘记了 重新 安装 同步



重新安装后 就可以了 

![image-20250308114446641](https://oss.caibucai.top/md/image-20250308114446641.png)

以上这种方式 适合 主程序相关的调试 是无法进入 依赖的 dll 进行调试的



### dll 项目 启用 选项

可以看到 我们无法单步进入了 ，原因是 release下 ,  dll 没有 生成pdb ，那该如何做，就是对 dll 项目 重复执行 这三步 

1. C/C++ --> 常规 调试信息格式设置为“程序[数据库(/Zi) 
2. C/C++ --> 优化 ，将右侧优化设置为 已禁用 
3. 选择链接器 --> 调试，将右侧生成调试信息 设置为 是

![image-20250308122023263](https://oss.caibucai.top/md/image-20250308122023263.png)



然后重新安装 就可以 调试 dll 项目的代码了

![image-20250308122141499](https://oss.caibucai.top/md/image-20250308122141499.png)

成功进入断住



## 附加到进程调试

这种场景 特别适用我们提供的代码 封装成dll 给别人调用，有问题调试时，别人把主程序发你 ，然后你就可以用这种方式进行调试



这里 我们 演示 一个插件调试的案例



首先我们要明确 我们有什么

1. 别人提供的 主程序 exe
2. 我们开发的 算法dll 工程 / 源码 也就是说 我们是可以生成 pdb 文件（dll 形式提供给别人用）

这种情况 不要从 vs 中运行项目，而是从安装目录下启动项目



启动后，可以看到 这里的pca 就是我们的 提供给别人的算法dll

![image-20250308140337682](https://oss.caibucai.top/md/image-20250308140337682.png)



我们的 dll 工程，并已经打上断点

![image-20250308140730726](https://oss.caibucai.top/md/image-20250308140730726.png)



附加到进程

![image-20250308140900355](https://oss.caibucai.top/md/image-20250308140900355.png)

找到我们运行的 exe 进程【从 vs 启动 这里是灰色 无法选中】

![image-20250308141008106](https://oss.caibucai.top/md/image-20250308141008106.png)

点击 附加 按钮，就启动了调试模式



然后点击前端 算法dll提供的按钮

![cpp_debug_attach_to_proccess](https://oss.caibucai.top/md/cpp_debug_attach_to_proccess.png)

可以看到 右侧成功断住



然后你就可以 快乐 debug了



## 总结

1. 无法直接启动成功 提示 dll 找不到，要么进行安装 将dll 和 exe 放在一起，要么手动调整 各个模块的 输出目录（没有实现自动化，感觉有点麻烦，如果有自动化方法 欢迎交流），并调整 调试目录

1. debug 模式下 自动生成 pdb , 但如果采用安装 记得 每次都要重新安装，否则会提示 符号未加载，无法命中断点
2. release 模式下 启用 三个选项 后 可以生成 pdb 然后重新安装更新，就可以正常调试了
3. 附加到进程调试， 特别适用于 仅dll 项目开发场景 调试，但是需要提供主程序，同时也需要pdb【好像原因是 生成的dll记录了dll源码的绝对路径，然后调试时 可以找到dll源码】
4. 注意！！！注意！！！注意！！！当提示 符号未加载，无法命中断点 ，很有可能就是 你修改代码 后 dll 和 pdb 不一致【特别是在 使用安装方式 情况下 不要慌 重新安装即可】

