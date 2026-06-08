import json
import re
from openai import OpenAI

# ==========================================
# 第一部分：配置模型接口 (已适配 DashScope 阿里云百炼)
# ==========================================
client = OpenAI(
    api_key="",  # 你的 DashScope API Key
    base_url=""  # 阿里云兼容接口地址
)
MODEL_NAME = "qwen-plus"  # 使用 qwen-plus 或 qwen-turbo


# ==========================================
# 第二部分：定义技能库 (Skill Library)
# ==========================================
class SkillLibrary:
    def __init__(self, filepath="paper_skills.json"):
        self.filepath = filepath
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.skills = json.load(f)
        except FileNotFoundError:
            self.skills = {}

    def add_skill(self, skill_name, description, code):
        self.skills[skill_name] = {
            "description": description,
            "code": code
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=4)
        print(f"[*] 技能 '{skill_name}' 已成功蒸馏并保存至本地科研技能库！")

    def get_skill(self, skill_name):
        return self.skills.get(skill_name, None)


# ==========================================
# 第三部分：定义科研智能体及其执行器
# ==========================================
class ResearchAgent:
    def __init__(self):
        self.skill_lib = SkillLibrary()

    def ask_llm(self, prompt):
        """与大模型交互的基础封装"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            # 【核心修复】修正新版本的提取语法：choices[0].message.content
            return response.choices[0].message.content
        except Exception as e:
            print(f"[!] API 调用失败: {e}")
            return None

    def distill_skill(self, task_description, skill_name):
        """阶段一：技能蒸馏"""
        print(f"\n--- 开始技能蒸馏：{skill_name} ---")
        prompt = f"""
你是一个精通Python和学术论文写作的科研助手。
现在有一个论文写作的常规任务：{task_description}
请编写一个Python函数来实现这个功能。
要求：
1. 函数名必须为 `{skill_name}`。
2. 函数必须接收一个参数 `data`（一个字典列表）。
3. 函数必须返回处理后的字符串（可以直接复制到论文中的内容）。
4. 请只返回完整的Python代码，使用 ```python 和 ``` 包含，不要任何解释。
"""
        response = self.ask_llm(prompt)
        if not response:
            print("[!] 模型未返回有效内容。")
            return None

        # 提取Markdown中的代码块
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            skill_code = code_match.group(1)
            self.skill_lib.add_skill(skill_name, task_description, skill_code)
            return skill_code
        else:
            print("[!] 代码提取失败，请检查模型输出。")
            print(f"模型原始返回内容：\n{response}")
            return None

    def execute_skill(self, skill_name, input_data):
        """阶段二：技能调用实践"""
        print(f"\n--- 开始调用技能：{skill_name} ---")
        skill = self.skill_lib.get_skill(skill_name)
        if not skill:
            print(f"[!] 技能库中未找到技能：{skill_name}")
            return

        print("[*] 从技能库检索到代码，正在通过Python解释器直接生成论文内容...")
        local_env = {}
        try:
            exec(skill["code"], globals(), local_env)
            func = local_env[skill_name]
            result = func(input_data)
            print("\n[*] 技能执行成功！生成的论文片段如下：\n")
            print(result)
            return result
        except Exception as e:
            print(f"[!] 执行技能时出错：{e}")


# ==========================================
# 第四部分：主实验流程 (学生操作区)
# ==========================================
if __name__ == "__main__":
    agent = ResearchAgent()

    task_desc = "根据输入的字典列表生成一个Markdown格式的三线表"
    skill_name = "generate_table"
    # 执行蒸馏（会联网调用通义千问生成新代码）
    agent.distill_skill(task_desc, skill_name)

    mock_data = [
        {"model": "ResNet50", "accuracy": "92.5%"},
        {"model": "VGG16", "accuracy": "90.1%"}
    ]
    # 执行技能（运行刚刚生成的真实代码）
    agent.execute_skill(skill_name, mock_data)