# 总结:🚀 **Git 快速指令手册（GitHub 协作）**


#### **一、初始化配置**
1. **安装 Git**（已装跳过）  
   ```bash
   brew install git
   ```

2. **设置用户名/邮箱**（全局）  
   ```bash
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```

3. **生成 SSH 密钥**（自定义路径示例）  
   ```bash
   ssh-keygen -t rsa -b 4096 -C "注释"  # 如：-C "macbook-work"
   # 保存路径：/Users/你/Desktop/gitkey/id_rsa（按提示回车）
   ```

4. **加载密钥到代理**  
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add /Users/你/Desktop/gitkey/id_rsa  # 替换实际路径
   ```

5. **复制公钥到 GitHub**  
   ```bash
   pbcopy < /Users/你/Desktop/gitkey/id_rsa.pub  # 粘贴到 GitHub 设置
   ```


#### **二、创建 & 关联仓库**
1. **本地初始化仓库（默认 main 分支）**  
   ```bash
   mkdir 项目名 && cd 项目名
   git init -b main
   ```

2. **关联远程仓库**  
   ```bash
   git remote add origin git@github.com:你的用户名/仓库名.git
   ```

3. **首次推送（解决远程已有文件冲突）**  
   ```bash
   git pull --rebase origin main  # 拉取远程（如 README）
   git push -u origin main       # 推送本地 main
   ```


#### **三、分支开发（单人）**
1. **创建并切换分支**  
   ```bash
   git switch -c 分支名  # 等价：git checkout -b 分支名
   ```

2. **提交代码**  
   ```bash
   git add .           # 添加所有改动
   git commit -m "描述提交"
   ```

3. **推送分支到远程**  
   ```bash
   git push -u origin 分支名  # 首次推送
   git push            # 后续推送
   ```


#### **四、协作合并（负责人操作）**
1. **同学推送分支并提 PR**  
   同学：`git push -u origin 他的分支名` → 在 GitHub 发起合并请求。

2. **你审核并合并**  
   - 解决冲突（本地）：  
     ```bash
     git fetch origin 他的分支名  # 拉取远程分支
     git switch main              # 切回主分支
     git merge origin/他的分支名  # 合并，手动改冲突文件
     git add 冲突文件 && git commit -m "合并说明"
     git push origin main         # 推送到远程
     ```
   - 直接在 GitHub PR 界面点击 **Merge**（推荐）。

3. **删除已合并分支**（可选）  
   ```bash
   git branch -d 他的分支名    # 删除本地分支
   git push origin --delete 他的分支名  # 删除远程分支
   ```


#### **五、常见问题**
1. **推送被拒（远程有更新）**  
   ```bash
   git pull --rebase origin main  # 先拉取合并，再推送
   ```

2. **合并冲突解决**  
   手动修改文件中 `<<<<<<<` 标记的冲突部分 → `git add 冲突文件` → `git commit`。


### 📌 **极简流程图**
```
本地开发：
新建分支 → 写代码 → add → commit → push（分支）

协作合并：
同学提 PR → 你审查/解决冲突 → 点击 Merge（GitHub）→ 删除废分支
```

**记住：永远不要直接在 main 分支开发！** 🔥  
（全文完，按场景查指令即可）