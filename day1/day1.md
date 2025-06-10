# 笔记

# Python与Git基础

## 1. 环境准备

## 2. 变量、变量类型、作用域

- 基本变量类型：`int`、`float`、`str`、`bool`、`list`、`tuple`、`dict`、`set`。
- 作用域：全局变量、局部变量，`global`和`nonlocal`关键字。
- 类型转换：如`int()`、`str()`。

## 3. 运算符及表达式

- 算术运算符：`+`, , , `/`, `//`, `%`, `*`。
- 比较运算符：`==`, `!=`, `>`, `<`, `>=`, `<=`。
- 逻辑运算符：`and`, `or`, `not`。
- 位运算符：`&`, `|`, `^`, `<<`, `>>`。

## 4. 语句：条件、循环、异常

- 条件语句：`if`, `elif`, `else`。
- 循环语句：`for`, `while`, `break`, `continue`。
- 异常处理：`try`, `except`, `finally`。

## 5. 函数：定义、参数、匿名函数、高阶函数

- 函数定义：`def`关键字，默认参数，可变参数（`args`, `*kwargs`）。
- 匿名函数：`lambda`。
- 高阶函数：接受函数作为参数或返回函数。

## 6. 包和模块：定义模块、导入模块、使用模块、第三方模块

- 模块：`import`语句，`from ... import ...`。
- 创建模块：一个`.py`文件。
- 包：包含`__init__.py`的文件夹。
- 第三方模块：如`requests`, `numpy`。

## 7. 类和对象

- 类定义：`class`关键字，属性和方法。
- 继承、多态、封装。
- 实例化对象。

## 8. 装饰器

- 装饰器本质：高阶函数，接受函数并返回新函数。
- 使用`@`语法。
- 带参数的装饰器。

## 9. 文件操作

- 读写文本文件：`open()`, `read()`, `write()`。
- 上下文管理器：`with`语句。
- 处理CSV、JSON文件。

# git命令操作

git init 初始化仓库

git add . 添加到暂存

git commit -m “” 

git remote add origin ”“

git pull --rebase origin main 

git push origin main

git config —global [user.name](http://user.name) “”

git config —global user.email

![image.png](image.png)

git remote add origin [https://github.com/Yunx-04/huaqing.git](https://github.com/Yunx-04/huaqing.git)

检查是否连接成功

![image.png](image%201.png)

![image.png](image%202.png)

conda命令

![屏幕截图 2025-06-09 143937.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_2025-06-09_143937.png)

![屏幕截图 2025-06-09 144004.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_2025-06-09_144004.png)

![屏幕截图 2025-06-09 165740.png](%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE_2025-06-09_165740.png)

远程代理提交Git