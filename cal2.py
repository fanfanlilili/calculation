import random
import argparse
from fractions import Fraction
import operator

# 添加 line_profiler 的装饰器，如果不存在则定义为空装饰器
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func


# --- 分数的自定义格式化 ---

@profile
def to_custom_format(f):
    """将 Fraction 对象或整数转换为指定的字符串格式。"""
    if isinstance(f, int) or f.denominator == 1:
        return str(int(f))

    # 移除了冗余的 Fraction(f) 转换，因为传入的已经是 Fraction 对象
    if f.numerator > f.denominator:
        integer_part = f.numerator // f.denominator
        remainder_numerator = f.numerator % f.denominator
        if remainder_numerator == 0:
            return str(integer_part)
        return f"{integer_part}'{remainder_numerator}/{f.denominator}"
    else:
        return f"{f.numerator}/{f.denominator}"


@profile
def from_custom_format(s):
    """将自定义格式的字符串转换为 Fraction 对象。"""
    s = s.strip()
    if "'" in s:
        integer_part, fraction_part = s.split("'")
        return Fraction(integer_part) + Fraction(fraction_part)
    elif "/" in s:
        return Fraction(s)
    else:
        return Fraction(int(s))


# --- AST (抽象语法树) 表示 ---

class ExpressionNode:
    """表示表达式树中的一个节点。"""

    def __init__(self, value, left=None, right=None):
        self.value = value  # 可以是 Fraction 对象或运算符字符串
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


# --- 表达式生成 ---

OPERATORS = ['+', '-', '×', '÷']
PRECEDENCE = {'+': 1, '-': 1, '×': 2, '÷': 2}


@profile
def generate_number(max_range):
    """在指定范围内生成一个随机自然数或真分数。"""
    if random.choice([True, False]):  # 50%的概率生成整数，50%的概率生成分数
        return Fraction(random.randint(0, max_range - 1))
    else:
        denominator = random.randint(2, max_range - 1)
        numerator = random.randint(1, denominator - 1)
        return Fraction(numerator, denominator)


@profile
def generate_expression_ast(max_range, ops_remaining):
    """递归生成一个有效的表达式AST。"""
    if ops_remaining == 0 or random.random() < 0.4:  # 基本情况或有概率直接生成一个数
        return ExpressionNode(generate_number(max_range))

    op = random.choice(OPERATORS)
    ops_for_children = ops_remaining - 1

    # 将剩余的运算符分配给子节点
    left_ops = random.randint(0, ops_for_children)
    right_ops = ops_for_children - left_ops

    # 限制重试次数，避免无限循环
    max_attempts = 20
    attempts = 0

    while attempts < max_attempts:  # 循环直到生成一个有效的表达式
        attempts += 1

        left_child = generate_expression_ast(max_range, left_ops)
        right_child = generate_expression_ast(max_range, right_ops)

        left_val = evaluate_ast(left_child)
        right_val = evaluate_ast(right_child)

        # 应用约束条件
        if op == '-' and left_val < right_val:
            continue  # 重试：结果会是负数

        if op == '÷':
            if right_val == 0:
                continue  # 重试：除以零
            result = left_val / right_val
            if result.denominator == 1:
                continue  # 重试：除法结果是整数（要求为真分数）

        # 如果所有约束都通过
        return ExpressionNode(op, left_child, right_child)

    # 如果多次尝试都失败，回退到简单的加法表达式
    return ExpressionNode('+',
                          generate_expression_ast(max_range, left_ops),
                          generate_expression_ast(max_range, right_ops))


# --- AST 处理 (求值、格式化、规范化) ---

@profile
def evaluate_ast(node):
    """对表达式树求值并返回一个 Fraction 对象。"""
    if node.is_leaf():
        return node.value

    left_val = evaluate_ast(node.left)
    right_val = evaluate_ast(node.right)

    op_map = {'+': operator.add, '-': operator.sub, '×': operator.mul, '÷': operator.truediv}
    return op_map[node.value](left_val, right_val)


@profile
def format_ast_to_string(node, parent_op=None):
    """将AST转换为带有正确括号的字符串。"""
    if node.is_leaf():
        return to_custom_format(node.value)

    op = node.value
    op_precedence = PRECEDENCE[op]

    # 递归格式化子节点
    left_str = format_ast_to_string(node.left, op)
    right_str = format_ast_to_string(node.right, op)

    # 如果需要，添加括号
    # 左子节点：如果其运算符的优先级较低
    if not node.left.is_leaf() and PRECEDENCE[node.left.value] < op_precedence:
        left_str = f"({left_str})"

    # 右子节点：如果其运算符的优先级较低或相等（对于-和÷）
    if not node.right.is_leaf():
        right_precedence = PRECEDENCE[node.right.value]
        if right_precedence < op_precedence:
            right_str = f"({right_str})"
        # 对于非交换运算符，相同的优先级也需要括号
        elif right_precedence == op_precedence and op in ['-', '÷']:
            right_str = f"({right_str})"

    res = f"{left_str} {op} {right_str}"

    # 如果当前表达式的优先级低于其父运算符，则为整个表达式添加括号
    if parent_op and op_precedence < PRECEDENCE[parent_op]:
        return f"({res})"

    return res


@profile
def get_canonical_form(node):
    """创建一个唯一的、排序过的AST表示，用于检测重复题目。"""
    if node.is_leaf():
        # 使用元组来表示分数，使其可哈希
        return (node.value.numerator, node.value.denominator)

    op = node.value
    left_canonical = get_canonical_form(node.left)
    right_canonical = get_canonical_form(node.right)

    # 对于可交换运算符（+、×），对操作数进行排序以确保唯一性
    if op in ['+', '×']:
        # 在排序前将所有项转为字符串进行比较，避免类型错误
        return (op, tuple(sorted((left_canonical, right_canonical), key=str)))
    else:  # 对于-和÷，顺序很重要
        return (op, left_canonical, right_canonical)


# --- 主应用逻辑 ---

@profile
def generate_problems(num_problems, max_range):
    """生成一组不重复的题目及其答案。"""
    print(f"正在生成 {num_problems} 道数值范围在 {max_range} 以内的不重复题目...")
    questions = []
    answers = []
    generated_forms = set()

    while len(questions) < num_problems:
        # 生成一个包含1到3个运算符的表达式
        num_ops = random.randint(1, 3)
        ast = generate_expression_ast(max_range, num_ops)

        canonical_form = get_canonical_form(ast)

        if canonical_form not in generated_forms:
            generated_forms.add(canonical_form)

            question_str = format_ast_to_string(ast) + " ="
            answer_val = evaluate_ast(ast)
            answer_str = to_custom_format(answer_val)

            questions.append(question_str)
            answers.append(answer_str)

            if len(questions) % 100 == 0 and len(questions) > 0:
                print(f"已生成 {len(questions)}/{num_problems} 道题目...")

    with open("Exercises.txt", "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            f.write(f"{i + 1}. {q}\n")

    with open("Answers.txt", "w", encoding="utf-8") as f:
        for i, a in enumerate(answers):
            f.write(f"{i + 1}. {a}\n")

    print("生成完毕！")
    print("题目已保存到 Exercises.txt")
    print("答案已保存到 Answers.txt")


@profile
def check_answers(exercise_file, answer_file):
    """比较习题文件和答案文件，并生成评分报告。"""
    try:
        with open(exercise_file, "r", encoding="utf-8") as f:
            exercises = f.readlines()
        with open(answer_file, "r", encoding="utf-8") as f:
            user_answers = f.readlines()
    except FileNotFoundError as e:
        print(f"错误: {e}。请检查文件路径。")
        return

    correct_indices = []
    wrong_indices = []

    for i, line in enumerate(exercises):
        # 清理行数据
        try:
            question_part = line.split('. ', 1)[1].strip()
            user_answer_part = user_answers[i].split('. ', 1)[1].strip()

            # 解析并计算正确答案
            expression_to_eval = question_part.replace(' =', '').strip()

            # 由于自定义格式的存在，直接用eval会有问题。
            # 需要一个安全的、能解析自定义格式的求值器。
            # 这里我们用一个包装函数来处理。
            def safe_eval_wrapper(expr_str):
                import re

                # 为了安全，先检查不允许的字符
                if re.search(r"[a-zA-Z_]{2,}", expr_str):
                    raise ValueError("Invalid characters in expression")

                # 替换运算符
                expr_str = expr_str.replace('×', '*').replace('÷', '/')

                # 将 x'y/z 格式转换为 Fraction()
                expr_str = re.sub(r"(\d+)'(\d+)/(\d+)", r'Fraction(\g<1>*\g<3>+\g<2>, \g<3>)', expr_str)
                # 将 y/z 格式转换为 Fraction()
                expr_str = re.sub(r"(\d+)/(\d+)", r'Fraction(\g<1>, \g<2>)', expr_str)
                # 将单独的整数转换为 Fraction()
                expr_str = re.sub(r"\b(\d+)\b(?!\s*')", lambda m: f"Fraction({m.group(1)})", expr_str)

                # 允许的命名空间
                allowed_names = {"Fraction": Fraction}
                # 使用 compile 来进一步限制
                code = compile(expr_str, "<string>", "eval")
                for name in code.co_names:
                    if name not in allowed_names:
                        raise NameError(f"Use of {name} is not allowed")

                return eval(code, {"__builtins__": None}, allowed_names)

            correct_val = safe_eval_wrapper(expression_to_eval)
            user_val = from_custom_format(user_answer_part)

            if user_val == correct_val:
                correct_indices.append(i + 1)
            else:
                wrong_indices.append(i + 1)

        except Exception as e:
            print(f"无法处理第 {i + 1} 行: {line.strip()}。错误: {e}")
            wrong_indices.append(i + 1)

    # 写入评分报告
    with open("Grade.txt", "w", encoding="utf-8") as f:
        f.write(f"Correct: {len(correct_indices)} ({', '.join(map(str, correct_indices))})\n")
        f.write(f"Wrong: {len(wrong_indices)} ({', '.join(map(str, wrong_indices))})\n")

    print("批改完毕！结果已保存到 Grade.txt")


@profile
def main():
    """主函数，用于解析参数并运行程序。"""
    parser = argparse.ArgumentParser(description="生成小学四则运算题目。", add_help=False)

    # 自定义帮助信息
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                        help="显示此帮助信息并退出。")

    parser.add_argument("-n", type=int, help="要生成的题目数量。")
    parser.add_argument("-r", type=int, help="题目中数值的范围（例如，10 表示数值小于10）。生成题目时此参数为必需。")

    parser.add_argument("-e", type=str, help="用于批改的习题文件路径。")
    parser.add_argument("-a", type=str, help="用于批改的答案文件路径。")

    args = parser.parse_args()

    if args.e and args.a:
        # 批改模式
        check_answers(args.e, args.a)
    elif args.n and args.r:
        # 生成模式
        if args.r < 2:
            print("错误：数值范围（-r）必须大于等于2，才能生成分数。")
        else:
            generate_problems(args.n, args.r)
    else:
        # 无效参数或缺少参数
        if not (args.e and args.a) and not (args.n and args.r):
            print("错误：参数不足或组合无效。")
            print("生成题目，请使用：python main.py -n <题目数量> -r <数值范围>")
            print("批改答案，请使用：python main.py -e <习题文件名.txt> -a <答案文件名.txt>")
            print("使用 -h 或 --help 查看帮助。")


if __name__ == "__main__":
    main()