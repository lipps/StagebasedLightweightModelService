#!/usr/bin/env python3
"""
生成模拟的 bs_train.jsonl 测试数据
用于验证 train_bs_logreg_bge.py 训练流程
"""

import json
import random
from pathlib import Path

random.seed(42)

# 正样本模板 (label=1) - 假设 BS 是 "商业策略/Business Strategy" 相关
POSITIVE_TEMPLATES = [
    # 商业策略相关
    "公司计划在下季度推出新的{product}产品线，预计能够{benefit}。",
    "根据市场分析，我们需要调整{aspect}策略以应对竞争对手的挑战。",
    "本次战略会议讨论了{topic}的实施方案和预期收益。",
    "为了提升市场份额，建议采取{strategy}的营销策略。",
    "财务报告显示，{metric}指标较上季度增长了{percent}%。",
    "竞争对手{competitor}最近发布的产品对我们构成了一定威胁。",
    "建议在{region}市场加大投入，开拓新的客户群体。",
    "本季度的销售目标是{amount}万元，需要各部门协同配合。",
    "根据用户反馈，产品的{feature}功能需要进一步优化。",
    "董事会批准了{budget}万元的预算用于市场推广活动。",
    "我们的核心竞争优势在于{advantage}，需要持续强化。",
    "行业报告预测，未来三年{industry}市场将保持{growth}%的增长率。",
    "建议与{partner}建立战略合作关系，实现资源互补。",
    "客户满意度调查显示，{satisfaction}%的用户对服务表示满意。",
    "为了降低运营成本，计划对{process}流程进行优化。",
    "新产品的定价策略需要考虑{factor}等多方面因素。",
    "品牌建设是长期投资，建议在{channel}渠道加强宣传。",
    "供应链管理的优化可以有效降低{cost}成本。",
    "数字化转型是当前企业发展的重要方向，建议优先推进{area}。",
    "人才是企业最重要的资产，需要完善{policy}机制。",
]

# 负样本模板 (label=0) - 非商业策略内容
NEGATIVE_TEMPLATES = [
    # 日常闲聊
    "今天天气真不错，适合出去走走。",
    "昨晚的电影很好看，推荐大家去看。",
    "周末打算去{place}玩，有人一起吗？",
    "最近在学习{skill}，感觉挺有意思的。",
    "这家餐厅的{food}做得很地道，下次再来。",
    # 技术讨论
    "这个 bug 是由于{cause}导致的，已经修复了。",
    "建议使用{framework}框架，性能更好。",
    "代码需要添加单元测试，确保质量。",
    "服务器的内存使用率过高，需要优化。",
    "数据库查询太慢，建议添加索引。",
    # 生活琐事
    "快递到了，麻烦帮我签收一下。",
    "下午三点有个会议，别忘了参加。",
    "空调温度调低一点，有点热。",
    "咖啡机坏了，需要联系维修。",
    "打印机没纸了，谁去补充一下？",
    # 新闻资讯
    "今日股市收盘，上证指数{change}。",
    "某地发生{event}，目前情况稳定。",
    "天气预报显示明天有{weather}。",
    "某明星宣布{news}，引发热议。",
    "世界杯比赛结果：{team}获胜。",
    # 学术技术
    "论文已经提交，等待审稿结果。",
    "实验数据显示{result}，需要进一步分析。",
    "算法的时间复杂度是O({complexity})。",
    "模型训练了{epochs}个epoch，效果不错。",
    "使用{method}方法可以提高准确率。",
]

# 填充词
PRODUCTS = ["智能家居", "云计算", "AI助手", "物联网", "移动支付", "在线教育", "健康管理"]
BENEFITS = ["提升用户体验", "增加收入来源", "扩大市场份额", "降低运营成本", "提高效率"]
ASPECTS = ["定价", "渠道", "产品", "服务", "营销", "品牌"]
TOPICS = ["数字化转型", "国际化拓展", "产品创新", "组织架构调整", "成本控制"]
STRATEGIES = ["差异化", "低成本", "聚焦细分", "多元化", "品牌升级"]
METRICS = ["营收", "利润", "用户数", "转化率", "留存率", "NPS得分"]
PERCENTS = ["5", "8", "12", "15", "20", "25", "30"]
COMPETITORS = ["A公司", "B集团", "C科技", "D互联网", "E控股"]
REGIONS = ["华东", "华南", "华北", "西南", "海外", "下沉市场"]
AMOUNTS = ["500", "800", "1000", "1500", "2000", "3000"]
FEATURES = ["搜索", "推荐", "支付", "社交", "直播", "客服"]
BUDGETS = ["100", "200", "300", "500", "800"]
ADVANTAGES = ["技术领先", "成本优势", "品牌影响力", "渠道覆盖", "用户粘性"]
INDUSTRIES = ["电商", "金融科技", "在线教育", "医疗健康", "智能制造"]
GROWTHS = ["10", "15", "20", "25", "30"]
PARTNERS = ["阿里", "腾讯", "华为", "字节", "美团"]
SATISFACTIONS = ["85", "88", "90", "92", "95"]
PROCESSES = ["采购", "生产", "物流", "销售", "售后"]
FACTORS = ["成本", "竞品价格", "用户承受力", "品牌定位"]
CHANNELS = ["社交媒体", "短视频", "搜索引擎", "线下活动"]
COSTS = ["库存", "物流", "人力", "采购"]
AREAS = ["客户管理", "财务系统", "供应链", "数据分析"]
POLICIES = ["激励", "培训", "晋升", "绩效考核"]

# 负样本填充词
PLACES = ["杭州", "上海", "北京", "深圳", "成都", "西湖", "黄山"]
SKILLS = ["Python", "摄影", "烹饪", "吉他", "画画", "游泳"]
FOODS = ["红烧肉", "小龙虾", "火锅", "寿司", "披萨"]
CAUSES = ["空指针", "内存泄漏", "并发问题", "配置错误", "网络超时"]
FRAMEWORKS = ["React", "Vue", "Spring", "Django", "FastAPI"]
CHANGES = ["上涨1.2%", "下跌0.8%", "持平", "微涨0.3%"]
EVENTS = ["地震", "暴雨", "交通事故", "演唱会"]
WEATHERS = ["小雨", "晴天", "多云", "大风"]
NEWS = ["新专辑发布", "结婚喜讯", "退役声明", "慈善捐款"]
TEAMS = ["巴西", "阿根廷", "法国", "德国"]
RESULTS = ["显著差异", "线性关系", "正相关", "无显著性"]
COMPLEXITIES = ["n", "nlogn", "n²", "logn"]
EPOCHS_LIST = ["10", "50", "100", "200"]
METHODS = ["集成学习", "数据增强", "迁移学习", "特征工程"]


def fill_template(template: str, is_positive: bool) -> str:
    """填充模板中的占位符"""
    result = template

    if is_positive:
        result = result.replace("{product}", random.choice(PRODUCTS))
        result = result.replace("{benefit}", random.choice(BENEFITS))
        result = result.replace("{aspect}", random.choice(ASPECTS))
        result = result.replace("{topic}", random.choice(TOPICS))
        result = result.replace("{strategy}", random.choice(STRATEGIES))
        result = result.replace("{metric}", random.choice(METRICS))
        result = result.replace("{percent}", random.choice(PERCENTS))
        result = result.replace("{competitor}", random.choice(COMPETITORS))
        result = result.replace("{region}", random.choice(REGIONS))
        result = result.replace("{amount}", random.choice(AMOUNTS))
        result = result.replace("{feature}", random.choice(FEATURES))
        result = result.replace("{budget}", random.choice(BUDGETS))
        result = result.replace("{advantage}", random.choice(ADVANTAGES))
        result = result.replace("{industry}", random.choice(INDUSTRIES))
        result = result.replace("{growth}", random.choice(GROWTHS))
        result = result.replace("{partner}", random.choice(PARTNERS))
        result = result.replace("{satisfaction}", random.choice(SATISFACTIONS))
        result = result.replace("{process}", random.choice(PROCESSES))
        result = result.replace("{factor}", random.choice(FACTORS))
        result = result.replace("{channel}", random.choice(CHANNELS))
        result = result.replace("{cost}", random.choice(COSTS))
        result = result.replace("{area}", random.choice(AREAS))
        result = result.replace("{policy}", random.choice(POLICIES))
    else:
        result = result.replace("{place}", random.choice(PLACES))
        result = result.replace("{skill}", random.choice(SKILLS))
        result = result.replace("{food}", random.choice(FOODS))
        result = result.replace("{cause}", random.choice(CAUSES))
        result = result.replace("{framework}", random.choice(FRAMEWORKS))
        result = result.replace("{change}", random.choice(CHANGES))
        result = result.replace("{event}", random.choice(EVENTS))
        result = result.replace("{weather}", random.choice(WEATHERS))
        result = result.replace("{news}", random.choice(NEWS))
        result = result.replace("{team}", random.choice(TEAMS))
        result = result.replace("{result}", random.choice(RESULTS))
        result = result.replace("{complexity}", random.choice(COMPLEXITIES))
        result = result.replace("{epochs}", random.choice(EPOCHS_LIST))
        result = result.replace("{method}", random.choice(METHODS))

    return result


def generate_sample(label: int) -> dict:
    """生成单条样本"""
    if label == 1:
        template = random.choice(POSITIVE_TEMPLATES)
        text = fill_template(template, is_positive=True)
    else:
        template = random.choice(NEGATIVE_TEMPLATES)
        text = fill_template(template, is_positive=False)

    return {"text": text, "label": label}


def main():
    output_path = Path("/opt/bge-m3/data/bs_train.jsonl")

    # 生成 1500 条数据 (正负样本比例 1:1)
    num_positive = 750
    num_negative = 750

    samples = []

    # 生成正样本
    for _ in range(num_positive):
        samples.append(generate_sample(1))

    # 生成负样本
    for _ in range(num_negative):
        samples.append(generate_sample(0))

    # 打乱顺序
    random.shuffle(samples)

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 生成 {len(samples)} 条样本")
    print(f"   正样本: {num_positive}")
    print(f"   负样本: {num_negative}")
    print(f"   保存至: {output_path}")

    # 显示几条样本
    print("\n📝 样本预览:")
    for i, sample in enumerate(samples[:5]):
        print(f"   [{sample['label']}] {sample['text'][:60]}...")


if __name__ == "__main__":
    main()
