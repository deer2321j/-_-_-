import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import category_encoders as ce
import matplotlib.pyplot as plt
import logging
import warnings
import os
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer

# 设置环境变量来抑制警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置tqdm全局显示
tqdm.pandas()

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log1p")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in greater_equal")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 安全计算AUC的函数
def safe_roc_auc_score(y_true, y_score):
    """安全计算ROC AUC分数，处理只有一个类别的情况"""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logging.warning(f"只有一个类别存在于标签中，无法计算AUC，返回0.5")
        return 0.5
    return roc_auc_score(y_true, y_score)


class BookRecommendationDataset(Dataset):
    """自定义数据集类，处理特征工程和时序特征"""

    def __init__(self, df, tokenizer, bert_model, device, is_train=True, cache_dir="./cache"):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.is_train = is_train
        self.cache_dir = cache_dir

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 关键修复：先检查text_encoder是否有效
        if self.tokenizer is None or self.bert_model is None:
            logging.warning("BERT模型未正确初始化，将使用预计算的特征")
            # 确保DataFrame中有title_embedding列
            if 'title_embedding' not in self.df.columns:
                logging.error("数据集中缺少title_embedding列，无法继续")
                raise ValueError("数据集中缺少title_embedding列")

        self.graph_data = self._build_interaction_graph()
        self.features, self.labels = self._prepare_features()

        # 检查标签分布
        if self.is_train and self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            logging.info(f"数据集标签分布: {label_distribution}")
            if len(unique) < 2:
                logging.warning("数据集只包含一个类别，可能导致模型训练问题")

    def _build_interaction_graph(self):
        """构建用户-图书交互图"""
        logging.info("  构建用户-图书交互图...")

        user_ids = self.df['user_id'].unique()
        book_ids = self.df['book_id'].unique()

        user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        book_mapping = {book_id: idx + len(user_ids) for idx, book_id in enumerate(book_ids)}

        # 将节点数据直接创建在目标设备上
        user_nodes = torch.tensor([user_mapping[user_id] for user_id in self.df['user_id']],
                                  dtype=torch.long, device=self.device)
        book_nodes = torch.tensor([book_mapping[book_id] for book_id in self.df['book_id']],
                                  dtype=torch.long, device=self.device)

        edge_index = torch.stack([user_nodes, book_nodes], dim=0)

        edge_attr_data = self.df[['borrow_duration', 'renewal_count', 'is_renewed']].copy()
        for col in edge_attr_data.columns:
            edge_attr_data[col] = pd.to_numeric(edge_attr_data[col], errors='coerce').fillna(0.0)

        edge_attr = torch.tensor(edge_attr_data.values, dtype=torch.float, device=self.device)
        total_nodes = len(user_mapping) + len(book_mapping)

        graph_data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_users=len(user_mapping),
            num_books=len(book_mapping),
            num_nodes=total_nodes
        )

        # 增强的索引检查
        max_user_index = edge_index[0].max().item() if edge_index.size(1) > 0 else -1
        max_book_index = edge_index[1].max().item() if edge_index.size(1) > 0 else -1

        if max_user_index >= len(user_mapping):
            logging.warning(f"用户索引超出范围: {max_user_index} >= {len(user_mapping)}")
            mask = edge_index[0] < len(user_mapping)
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

        if max_book_index >= (len(book_mapping) + len(user_mapping)):
            logging.warning(f"图书索引超出范围: {max_book_index} >= {len(book_mapping) + len(user_mapping)}")
            mask = edge_index[1] < (len(book_mapping) + len(user_mapping))
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

        if edge_index.size(1) == 0:
            logging.warning("所有边都被过滤掉了，这会导致模型无法训练")

        return graph_data

    def _get_bert_embedding(self, text):
        """获取文本的嵌入向量 - 修复版本"""
        if pd.isna(text) or text == "":
            return np.zeros(768)

        # 关键修复：检查BERT模型是否可用
        if self.bert_model is None or self.tokenizer is None:
            logging.warning("BERT模型不可用，返回零向量")
            return np.zeros(768)

        try:
            inputs = self.tokenizer(
                str(text),
                return_tensors="pt",
                max_length=32,
                truncation=True,
                padding='max_length'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except Exception as e:
            logging.error(f"文本编码出错: {str(e)}")
            return np.zeros(768)  # 返回零向量而不是抛出异常

    def _prepare_features(self):
        """准备特征 - 修复版本，使用预计算的特征"""
        logging.info("  准备特征...")

        # 用户特征（4维：3个统计特征 + department编码）
        user_ids = self.df['user_id'].unique()
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

        # 使用向量化操作替代循环
        user_groups = self.df.groupby('user_id')
        user_features = user_groups.agg(
            avg_duration=('borrow_duration', 'mean'),
            total_borrows=('book_id', 'count'),
            renewal_rate=('is_renewed', 'mean'),
            department_encoded=('department_encoded', 'first')
        ).reset_index(drop=True)
        user_features = user_features.fillna(0).values

        # 图书特征 - 使用预计算的特征
        book_ids = self.df['book_id'].unique()
        book_id_to_idx = {book_id: idx for idx, book_id in enumerate(book_ids)}

        # 生成缓存文件名
        import hashlib
        book_ids_str = "_".join(map(str, sorted(book_ids)))
        hash_obj = hashlib.md5(book_ids_str.encode())
        cache_filename = f"book_features_{hash_obj.hexdigest()}.pkl"
        self.cache_path = os.path.join(self.cache_dir, cache_filename)

        # 检查缓存是否存在
        if os.path.exists(self.cache_path):
            logging.info(f"  加载预计算的图书特征缓存: {self.cache_path}")
            import pickle
            with open(self.cache_path, 'rb') as f:
                book_id_to_features = pickle.load(f)
        else:
            # 预计算图书特征 - 修复版本
            book_id_to_features = {}
            for book_id in tqdm(book_ids, desc="  预计算图书特征"):
                book_data = self.df[self.df['book_id'] == book_id]
                avg_duration = book_data['borrow_duration'].mean()
                total_borrows = len(book_data)
                renewal_rate = book_data['is_renewed'].mean()
                douban_rating = book_data['douban_rating'].iloc[0] if 'douban_rating' in book_data.columns else 0.0

                # 关键修复：直接使用预计算的title_embedding
                title_embedding = book_data['title_embedding'].iloc[
                    0] if 'title_embedding' in book_data.columns else np.zeros(768)

                # 确保title_embedding是numpy数组
                if not isinstance(title_embedding, np.ndarray):
                    if isinstance(title_embedding, list):
                        title_embedding = np.array(title_embedding)
                    else:
                        title_embedding = np.zeros(768)

                # 图书特征: 4个统计特征 + 题名嵌入(768维) = 772维
                book_id_to_features[book_id] = [avg_duration, total_borrows, renewal_rate, douban_rating,
                                                *title_embedding]

            # 保存特征缓存
            logging.info(f"  保存图书特征缓存到: {self.cache_path}")
            import pickle
            with open(self.cache_path, 'wb') as f:
                pickle.dump(book_id_to_features, f)

        # 组合特征
        features = []
        labels = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="  组合特征"):
            user_idx = user_id_to_idx[row['user_id']]
            book_id = row['book_id']

            user_feat = user_features[user_idx]
            book_feat = book_id_to_features[book_id]

            # 交互特征（4维）
            duration = row['borrow_duration']
            renewal = row['is_renewed']
            renewal_count = row['renewal_count']
            last_borrow_days = row['days_since_last_borrow']

            feature_vec = np.concatenate([user_feat, book_feat,
                                          [duration, renewal, renewal_count, last_borrow_days]])
            features.append(feature_vec)

            if self.is_train:
                labels.append(row['will_borrow_again'])

        return np.array(features), np.array(labels) if self.is_train else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_train:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

    def get_graph_data(self):
        return self.graph_data


class GNNModel(nn.Module):
    """可配置的图神经网络模型，支持GraphSAGE和GAT"""

    def __init__(self, num_user_features, num_book_features,
                 gnn_type='sage',
                 hidden_dim=64,
                 num_layers=2,
                 heads=1,
                 dropout=0.2):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 警告标志位（每个epoch仅打印1次节点不匹配警告）
        self.warned_node_mismatch = False

        # 用户和图书嵌入网络
        self.user_embedding = nn.Sequential(
            nn.Linear(num_user_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.book_embedding = nn.Sequential(
            nn.Linear(num_book_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 创建GNN层
        self.convs = nn.ModuleList()

        if gnn_type.lower() == 'sage':
            # GraphSAGE架构
            for i in range(num_layers):
                in_channels = hidden_dim if i > 0 else hidden_dim
                self.convs.append(SAGEConv(in_channels, hidden_dim))
        elif gnn_type.lower() == 'gat':
            # GAT架构
            for i in range(num_layers):
                in_channels = hidden_dim * heads if i > 0 else hidden_dim
                out_channels = hidden_dim if i == num_layers - 1 else hidden_dim
                self.convs.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout))
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")

        # 批归一化层
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type.lower() == 'gat':
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            else:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, graph_data, user_features, book_features):
        # 用户和图书嵌入
        user_emb = self.user_embedding(user_features)
        book_emb = self.book_embedding(book_features)
        x = torch.cat([user_emb, book_emb], dim=0)

        # 检查节点数量匹配
        if x.size(0) != graph_data.num_nodes:
            if not self.warned_node_mismatch and self.training:
                logging.warning(f"节点特征数量 {x.size(0)} 与图节点数量 {graph_data.num_nodes} 不匹配")
                self.warned_node_mismatch = True
            # 调整大小以匹配，防止索引错误
            if x.size(0) < graph_data.num_nodes:
                padding = torch.zeros(graph_data.num_nodes - x.size(0), x.size(1)).to(x.device)
                x = torch.cat([x, padding], dim=0)
            else:
                x = x[:graph_data.num_nodes, :]

        # 检查边索引有效性
        if graph_data.edge_index.numel() > 0:
            max_idx = graph_data.edge_index.max().item()
            if max_idx >= x.size(0) and not hasattr(self, 'warned_edge_idx'):
                logging.warning(f"边索引 {max_idx} 超出节点数量 {x.size(0)}，将过滤无效边")
                setattr(self, 'warned_edge_idx', True)
                mask = (graph_data.edge_index[0] < x.size(0)) & (graph_data.edge_index[1] < x.size(0))
                graph_data.edge_index = graph_data.edge_index[:, mask]
                if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                    graph_data.edge_attr = graph_data.edge_attr[mask]

        # 应用GNN层
        for i in range(self.num_layers):
            x = self.convs[i](x, graph_data.edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        return x


class BookRecommendationModel(nn.Module):
    """完整的推荐系统模型，修复特征维度不匹配问题"""

    def __init__(self, num_features, graph_model,
                 hidden_dim=64,
                 fusion_layers=2,
                 predictor_layers=2,
                 dropout=0.2):
        super().__init__()
        self.graph_model = graph_model
        self.hidden_dim = hidden_dim

        # 警告标志位
        self.warned_dim_mismatch = False

        # 时序特征处理
        self.temporal_module = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 重新计算融合层输入维度
        fusion_input_dim = (num_features - 4) + (hidden_dim // 2) + hidden_dim + hidden_dim

        logging.info(f"融合层输入维度计算: (总特征{num_features} - 时序特征4) + "
                     f"时序处理后{hidden_dim // 2} + 用户图嵌入{hidden_dim} + 图书图嵌入{hidden_dim} = "
                     f"总计{fusion_input_dim}")

        # 特征融合层
        fusion_layers_list = []
        for i in range(fusion_layers):
            in_dim = fusion_input_dim if i == 0 else hidden_dim * 2
            out_dim = hidden_dim * 2 if i < fusion_layers - 1 else hidden_dim
            fusion_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.fusion_layer = nn.Sequential(*fusion_layers_list)

        # 预测层
        predictor_layers_list = []
        for i in range(predictor_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim // 2
            out_dim = hidden_dim // 2 if i < predictor_layers - 1 else 1
            predictor_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < predictor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity()
            ])

        self.predictor = nn.Sequential(*predictor_layers_list)

    def forward(self, features, graph_data, user_features, book_features):
        # 时序特征处理
        temporal_features = features[:, -4:]
        temporal_emb = self.temporal_module(temporal_features)

        # 图神经网络处理
        graph_emb = self.graph_model(graph_data, user_features, book_features)

        # 提取用户和图书的图嵌入
        user_graph_emb = graph_emb[:graph_data.num_users]
        book_graph_emb = graph_emb[graph_data.num_users:]

        # 确保嵌入维度与批次大小匹配
        batch_size = features.size(0)

        # 安全地调整用户嵌入大小
        if len(user_graph_emb) == 0:
            user_graph_emb = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        elif user_graph_emb.size(0) < batch_size:
            repeat_factor = (batch_size // user_graph_emb.size(0)) + 1
            user_graph_emb = user_graph_emb.repeat(repeat_factor, 1)[:batch_size]
        else:
            user_graph_emb = user_graph_emb[:batch_size]

        # 安全地调整图书嵌入大小
        if len(book_graph_emb) == 0:
            book_graph_emb = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        elif book_graph_emb.size(0) < batch_size:
            repeat_factor = (batch_size // book_graph_emb.size(0)) + 1
            book_graph_emb = book_graph_emb.repeat(repeat_factor, 1)[:batch_size]
        else:
            book_graph_emb = book_graph_emb[:batch_size]

        # 组合所有特征
        combined = torch.cat([
            features[:, :-4],
            temporal_emb,
            user_graph_emb,
            book_graph_emb
        ], dim=1)

        # 检查组合特征维度是否与融合层输入维度匹配
        if combined.size(1) != self.fusion_layer[0].in_features:
            if not self.warned_dim_mismatch and self.training:
                logging.error(
                    f"特征维度不匹配: 组合特征{combined.size(1)} != 融合层输入{self.fusion_layer[0].in_features}")
                self.warned_dim_mismatch = True
            # 尝试动态调整以避免崩溃
            combined = combined[:, :self.fusion_layer[0].in_features] if combined.size(1) > self.fusion_layer[
                0].in_features else \
                torch.cat([combined, torch.zeros(batch_size, self.fusion_layer[0].in_features - combined.size(1)).to(
                    combined.device)], dim=1)

        # 特征融合与预测
        fused = self.fusion_layer(combined)
        return self.predictor(fused)


class BookRecommendationSystem:
    def __init__(self, data_path, douban_data_path, bert_model_name, mode='train',
                 test_size=0.2, sample_size=None, cache_dir="./cache"):
        self.data_path = data_path
        self.douban_data_path = douban_data_path
        self.bert_model_name = bert_model_name
        self.mode = mode
        self.test_size = test_size
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        self.df = None
        self.douban_df = None
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.graph_data = None
        self.model = None
        self.gnn_model = None
        self.text_encoder = None
        self.target_encoder = None
        self.scaler = None

        # 动态选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用 {'GPU' if torch.cuda.is_available() else 'CPU'} 进行计算")

        self.cat_features = []
        self.num_features = []

        self.load_data()

        # 测试模式：使用少量数据
        if self.mode == 'test' and self.sample_size is not None and len(self.df) > self.sample_size:
            if 'will_borrow_again' in self.df.columns:
                self.df = self.df.groupby('will_borrow_again', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), self.sample_size // 2))
                )
            else:
                self.df = self.df.sample(self.sample_size, random_state=42)
            logging.info(f"测试模式：使用 {len(self.df)} 条样本数据")

    def load_data(self):
        """加载并预处理数据"""
        logging.info("加载数据...")
        try:
            self.df = pd.read_csv(self.data_path, engine='python')
            logging.info(f"数据加载成功，共 {len(self.df)} 条记录")

            self.douban_df = pd.read_csv(self.douban_data_path, engine='python')
            logging.info(f"豆瓣数据加载成功，共 {len(self.douban_df)} 条记录")

            # 合并豆瓣评分到主数据
            self.df = pd.merge(self.df, self.douban_df[['book_id', 'douban_rating']],
                               on='book_id', how='left')
            logging.info(f"合并豆瓣评分后，共 {len(self.df)} 条记录")

        except Exception as e:
            logging.error(f"加载数据失败: {str(e)}")
            try:
                logging.info("尝试使用不同的编码方式加载数据...")
                self.df = pd.read_csv(self.data_path, engine='python', encoding='latin-1')
                self.douban_df = pd.read_csv(self.douban_data_path, engine='python', encoding='latin-1')
                self.df = pd.merge(self.df, self.douban_df[['book_id', 'douban_rating']],
                                   on='book_id', how='left')
                logging.info(f"使用latin-1编码加载数据成功，共 {len(self.df)} 条记录")
            except Exception as e2:
                logging.error(f"所有加载数据尝试都失败: {str(e2)}")
                raise

        required_columns = ['user_id', 'book_id', '借阅时间', 'department']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        # 仅保留必要的列以减少内存占用
        keep_columns = required_columns + ['还书时间', '续借时间', '续借次数', '是否续借',
                                           '一级分类', '题名', 'douban_rating']
        keep_columns = [col for col in keep_columns if col in self.df.columns]
        self.df = self.df[keep_columns].dropna(subset=required_columns)

        date_cols = ['借阅时间', '还书时间', '续借时间']
        for col in tqdm(date_cols, desc="  转换日期格式"):
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce', format='%Y-%m-%d %H:%M:%S')
                except:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            else:
                logging.warning(f"列 '{col}' 不存在于数据中")

        logging.info(f"数据集加载完成，共 {len(self.df)} 条记录")

    def prepare_features(self):
        """准备特征工程"""
        logging.info("开始特征工程...")

        # 1. 创建目标变量
        logging.info("步骤1/5: 创建目标变量...")
        self._create_target_variable()

        # 检查目标变量分布
        if 'will_borrow_again' in self.df.columns:
            target_dist = self.df['will_borrow_again'].value_counts(normalize=True)
            logging.info(f"目标变量分布: {target_dist.to_dict()}")
            if len(target_dist) < 2:
                logging.warning("目标变量只包含一个类别，可能导致模型训练问题")

        # 2. 创建基础特征
        logging.info("步骤2/5: 创建基础特征...")
        self._create_basic_features()

        # 3. 处理豆瓣评分特征
        logging.info("步骤3/5: 处理豆瓣评分特征...")
        self._process_douban_ratings()

        # 4. 文本特征编码 - 关键修复
        logging.info("步骤4/5: 处理文本特征...")
        self._encode_text_features()

        # 5. 类别特征编码
        logging.info("步骤5/5: 类别特征编码...")
        self._encode_categorical_features()

        # 6. 数值特征处理
        logging.info("步骤6/5: 数值特征处理...")
        self._process_numerical_features()

        # 创建数据集
        self._create_dataset()

        logging.info("特征工程完成")

    def _process_douban_ratings(self):
        """处理豆瓣评分特征"""
        if 'douban_rating' not in self.df.columns:
            logging.warning("数据中没有豆瓣评分列，将使用默认值0")
            self.df['douban_rating'] = 0.0
            return

        # 处理缺失值和0值
        original_zero_count = (self.df['douban_rating'] == 0).sum()
        logging.info(f"豆瓣评分为0的记录数: {original_zero_count}")

        # 计算非零评分的平均值
        non_zero_ratings = self.df[self.df['douban_rating'] > 0]['douban_rating']
        if len(non_zero_ratings) > 0:
            avg_rating = non_zero_ratings.mean()
            logging.info(f"非零豆瓣评分的平均值: {avg_rating:.2f}")

            # 将0值替换为平均分
            self.df.loc[self.df['douban_rating'] == 0, 'douban_rating'] = avg_rating
            logging.info(f"已将 {original_zero_count} 条记录的豆瓣评分从0替换为平均值 {avg_rating:.2f}")
        else:
            logging.warning("没有有效的豆瓣评分数据，所有评分将保持为0")

        # 检查是否有NaN值并填充
        nan_count = self.df['douban_rating'].isna().sum()
        if nan_count > 0:
            avg_rating = self.df['douban_rating'].mean()
            if pd.isna(avg_rating):
                avg_rating = 0.0
            self.df = self.df.assign(douban_rating=self.df['douban_rating'].fillna(avg_rating))
            logging.info(f"已将 {nan_count} 条记录的NaN豆瓣评分替换为平均值 {avg_rating:.2f}")

    def _create_target_variable(self):
        """创建目标变量：是否会二次借阅"""
        # 修复目标变量创建逻辑
        borrow_counts = self.df.groupby(['user_id', 'book_id']).size().reset_index(name='borrow_count')

        # 标记哪些记录是首次借阅
        self.df = self.df.sort_values(['user_id', 'book_id', '借阅时间'])
        self.df['is_first_borrow'] = ~self.df.duplicated(subset=['user_id', 'book_id'], keep='first')

        # 合并借阅次数信息
        self.df = pd.merge(self.df, borrow_counts, on=['user_id', 'book_id'], how='left')

        # 创建目标变量：首次借阅且借阅次数大于1的记录
        self.df['will_borrow_again'] = 0
        self.df.loc[(self.df['is_first_borrow']) & (self.df['borrow_count'] > 1), 'will_borrow_again'] = 1

        # 仅保留首次借阅记录用于训练
        self.df = self.df[self.df['is_first_borrow']].copy()

        # 检查目标变量分布
        pos_count = self.df['will_borrow_again'].sum()
        neg_count = len(self.df) - pos_count
        logging.info(f"目标变量分布: 正样本 {pos_count} ({pos_count / len(self.df) * 100:.2f}%), "
                     f"负样本 {neg_count} ({neg_count / len(self.df) * 100:.2f}%)")

    def _create_basic_features(self):
        """创建基础特征"""
        # 借阅时长
        if '还书时间' in self.df.columns and '借阅时间' in self.df.columns:
            duration = (self.df['还书时间'] - self.df['借阅时间']).dt.total_seconds()
            duration = duration.clip(lower=0)
            self.df['borrow_duration'] = duration / (24 * 3600)
            self.df['borrow_duration'] = self.df['borrow_duration'].fillna(0)
            self.df['log_borrow_duration'] = np.log1p(self.df['borrow_duration'].clip(lower=0))
            self.df['log_borrow_duration'] = self.df['log_borrow_duration'].replace([np.inf, -np.inf], 0)
        else:
            self.df['borrow_duration'] = 0
            self.df['log_borrow_duration'] = 0

        # 续借特征
        self.df['renewal_count'] = self.df.get('续借次数', 0).fillna(0).astype(float)
        self.df['is_renewed'] = self.df.get('是否续借', 0).fillna(0).astype(int)

        # 其他特征
        denominator = self.df['borrow_duration'] + 1e-5
        denominator = denominator.replace(0, 1)
        self.df['renewal_intensity'] = self.df['renewal_count'] / denominator
        self.df['is_industrial'] = (self.df.get('一级分类', '') == '工业技术').astype(int)

        # 用户统计特征
        self.df['user_borrow_count'] = self.df.groupby('user_id')['book_id'].transform('count')
        self.df['user_avg_duration'] = self.df.groupby('user_id')['borrow_duration'].transform('mean')
        self.df['user_renewal_rate'] = self.df.groupby('user_id')['is_renewed'].transform('mean')

        # 图书统计特征
        self.df['book_borrow_count'] = self.df.groupby('book_id')['user_id'].transform('count')
        self.df['book_avg_duration'] = self.df.groupby('book_id')['borrow_duration'].transform('mean')
        self.df['book_renewal_rate'] = self.df.groupby('book_id')['is_renewed'].transform('mean')

        # 时序特征
        self.df = self.df.sort_values(['user_id', '借阅时间'])
        self.df['days_since_last_borrow'] = self.df.groupby('user_id')['借阅时间'].diff().dt.days.fillna(0).clip(
            lower=0)

    def _encode_text_features(self):
        """处理文本特征 - 关键修复版本"""
        try:
            logging.info(f"尝试从Hugging Face Hub加载BERT模型: {self.bert_model_name}")

            # 尝试加载BERT模型
            tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model_name,
                trust_remote_code=True
            )

            model = AutoModel.from_pretrained(
                self.bert_model_name,
                trust_remote_code=True
            ).to(self.device)

            model.eval()

            # 使用BERT编码文本
            if '题名' in self.df.columns:
                logging.info("使用BERT编码文本特征...")
                self.df['title_embedding'] = self.df['题名'].progress_apply(
                    lambda x: self._get_bert_embedding_with_model(x, tokenizer, model)
                )
            else:
                logging.warning("数据中没有'题名'列，使用零向量替代")
                self.df['title_embedding'] = self.df.apply(lambda x: np.zeros(768), axis=1)

            self.text_encoder = (tokenizer, model)
            logging.info("BERT模型加载和文本编码完成")

        except Exception as e:
            logging.error(f"BERT模型加载失败: {str(e)}")
            logging.info("使用备用文本编码方案...")
            self._encode_text_features_fallback()

    def _get_bert_embedding_with_model(self, text, tokenizer, model):
        """使用指定的BERT模型获取文本嵌入"""
        try:
            if pd.isna(text) or text == "":
                return np.zeros(768)
            inputs = tokenizer(
                str(text),
                return_tensors="pt",
                max_length=32,
                truncation=True,
                padding='max_length'
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)

            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except Exception as e:
            logging.error(f"文本编码出错: {str(e)}")
            return np.zeros(768)

    def _encode_text_features_fallback(self):
        """备用文本编码方案"""
        try:
            logging.info("使用TF-IDF作为备用文本编码方案...")

            if '题名' in self.df.columns:
                # 使用简单的词频统计作为备用
                from sklearn.feature_extraction.text import TfidfVectorizer

                # 处理空值
                titles = self.df['题名'].fillna('')

                # 使用TF-IDF
                vectorizer = TfidfVectorizer(max_features=768, stop_words=None)
                title_vectors = vectorizer.fit_transform(titles)

                # 转换为numpy数组
                title_embeddings = title_vectors.toarray()

                # 如果维度不够768，用0填充
                if title_embeddings.shape[1] < 768:
                    padded_embeddings = np.zeros((len(title_embeddings), 768))
                    padded_embeddings[:, :title_embeddings.shape[1]] = title_embeddings
                    self.df['title_embedding'] = list(padded_embeddings)
                else:
                    self.df['title_embedding'] = list(title_embeddings[:, :768])

                logging.info(f"TF-IDF编码完成，特征维度: {title_embeddings.shape}")
            else:
                logging.warning("数据中没有'题名'列，使用零向量")
                self.df['title_embedding'] = self.df.apply(lambda x: np.zeros(768), axis=1)

            # 设置text_encoder为None，但确保title_embedding列存在
            self.text_encoder = (None, None)
            logging.info("备用文本编码方案完成")

        except Exception as e:
            logging.error(f"备用文本编码方案失败: {str(e)}")
            logging.info("使用零向量作为文本嵌入")
            self.df['title_embedding'] = self.df.apply(lambda x: np.zeros(768), axis=1)
            self.text_encoder = (None, None)

    def _encode_categorical_features(self):
        """编码类别特征"""
        possible_cat_features = ['一级分类', '二级分类', '出版社', 'department', 'user_type', 'gender']
        self.cat_features = [col for col in possible_cat_features if col in self.df.columns]

        if not self.cat_features:
            logging.warning("没有找到类别特征列，跳过类别编码")
            self.target_encoder = None
            return

        # 对department特征进行目标编码
        if 'department' in self.df.columns:
            dept_counts = self.df['department'].value_counts()
            logging.info(f"院系分布: {dept_counts.to_dict()}")

            self.target_encoder = ce.TargetEncoder(cols=['department'])
            self.target_encoder.fit(self.df[['department']], self.df['will_borrow_again'])
            encoded_department = self.target_encoder.transform(self.df[['department']])
            self.df['department_encoded'] = encoded_department['department']

            self.df = self.df.drop('department', axis=1)
            self.cat_features.remove('department')

        # 对其他类别特征进行编码
        if self.cat_features:
            self.target_encoder = ce.TargetEncoder(cols=self.cat_features)
            self.target_encoder.fit(self.df[self.cat_features], self.df['will_borrow_again'])
            encoded_features = self.target_encoder.transform(self.df[self.cat_features])
            for col in self.cat_features:
                self.df[f'{col}_encoded'] = encoded_features[col]

    def _process_numerical_features(self):
        """标准化数值特征"""
        base_num_features = [
            'borrow_duration', 'log_borrow_duration', 'renewal_count',
            'renewal_intensity', 'user_avg_duration', 'user_borrow_count',
            'user_renewal_rate', 'book_avg_duration', 'book_borrow_count',
            'book_renewal_rate', 'is_industrial', 'days_since_last_borrow',
            'douban_rating'
        ]

        if 'department_encoded' in self.df.columns:
            base_num_features.append('department_encoded')

        self.num_features = base_num_features + [f'{col}_encoded' for col in self.cat_features]
        self.num_features = [col for col in self.num_features if col in self.df.columns]

        if not self.num_features:
            logging.warning("没有数值特征可用于标准化")
            self.scaler = None
            return

        for col in tqdm(self.num_features, desc="  填充空值"):
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[self.num_features])
        for i, col in enumerate(self.num_features):
            self.df[col] = scaled_features[:, i]

    def _create_dataset(self):
        """创建数据集"""
        # 关键修复：确保text_encoder存在
        if self.text_encoder is None:
            # 如果text_encoder为None，创建虚拟的
            self.text_encoder = (None, None)
            logging.warning("text_encoder为None，使用虚拟值创建数据集")

        tokenizer, bert_model = self.text_encoder

        logging.info("创建数据集...")
        self.dataset = BookRecommendationDataset(
            self.df, tokenizer, bert_model, self.device,
            is_train=True, cache_dir=self.cache_dir
        )

        # 训练模式：划分训练集和测试集
        if self.mode == 'train':
            labels = self.df['will_borrow_again'].values
            train_indices, test_indices = train_test_split(
                range(len(self.dataset)),
                test_size=self.test_size,
                random_state=42,
                stratify=labels
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.test_dataset = Subset(self.dataset, test_indices)
            logging.info(f"数据集划分: 训练集 {len(self.train_dataset)} 条, 测试集 {len(self.test_dataset)} 条")
        elif self.mode == 'predict':
            self.train_dataset = self.dataset
            logging.info(f"预测模式: 使用全部 {len(self.train_dataset)} 条数据训练")
        elif self.mode == 'test':
            self.train_dataset = self.dataset
            logging.info(f"测试模式: 使用 {len(self.train_dataset)} 条数据训练")

        self.graph_data = self.dataset.get_graph_data()

    def train_model(self, epochs=10, lr=0.001, batch_size=16, patience=5,
                    pretrained_model_path=None, use_amp=True,
                    gnn_type='sage',
                    gnn_hidden_dim=64,
                    gnn_num_layers=2,
                    gnn_heads=1,
                    gnn_dropout=0.2,
                    model_hidden_dim=64,
                    model_fusion_layers=2,
                    model_predictor_layers=2,
                    model_dropout=0.2,
                    pos_weight=2.0):
        """训练模型"""
        logging.info("开始训练模型...")

        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        ) if self.test_dataset else None

        num_features = self.dataset.features.shape[1]
        logging.info(f"输入特征维度: {num_features}")

        # 图书特征维度：4个统计特征 + 文本嵌入维度(768) = 772维
        book_feat_dim = 4 + 768

        # 用户特征维度：3个统计特征 + department编码 = 4维
        user_feat_dim = 4

        # 创建GNN模型
        self.gnn_model = GNNModel(
            num_user_features=user_feat_dim,
            num_book_features=book_feat_dim,
            gnn_type=gnn_type,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            dropout=gnn_dropout
        ).to(self.device)

        # 创建推荐模型
        self.model = BookRecommendationModel(
            num_features=num_features,
            graph_model=self.gnn_model,
            hidden_dim=model_hidden_dim,
            fusion_layers=model_fusion_layers,
            predictor_layers=model_predictor_layers,
            dropout=model_dropout
        ).to(self.device)

        # 如果提供了预训练模型路径，则加载模型参数
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                logging.info(f"从预训练模型 {pretrained_model_path} 加载参数...")
                self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
                logging.info("预训练模型参数加载成功")
            except Exception as e:
                logging.error(f"加载预训练模型失败: {str(e)}，将从头开始训练")

        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )

        # 处理类别不平衡
        if pos_weight is None and hasattr(self, 'df') and 'will_borrow_again' in self.df.columns:
            pos_count = self.df['will_borrow_again'].sum()
            neg_count = len(self.df) - pos_count
            pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)]).to(self.device)
            logging.info(f"自动计算正样本权重: {pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([pos_weight]).to(self.device)
            logging.info(f"使用指定的正样本权重: {pos_weight.item():.2f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 混合精度训练
        scaler = GradScaler('cuda', enabled=use_amp and self.device.type == 'cuda')

        # 训练记录
        train_losses = []
        test_losses = []
        train_aucs = []
        test_aucs = []
        train_f1s = []
        test_f1s = []
        best_auc = 0
        best_f1 = 0
        no_improve = 0
        early_stop = False

        for epoch in range(epochs):
            if early_stop:
                logging.info(f"早停: 在 {epoch} 轮停止训练")
                break

            # 重置警告标志位
            self.gnn_model.warned_node_mismatch = False
            self.model.warned_dim_mismatch = False
            if hasattr(self.gnn_model, 'warned_edge_idx'):
                delattr(self.gnn_model, 'warned_edge_idx')

            self.model.train()
            epoch_train_loss = 0
            all_train_preds = []
            all_train_labels = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - 训练"):
                features, labels = batch
                features = features.float().to(self.device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(self.device, non_blocking=True)

                # 确保特征和标签没有NaN值
                features = torch.nan_to_num(features)
                labels = torch.nan_to_num(labels)

                user_features = features[:, :user_feat_dim]
                book_features = features[:, user_feat_dim:user_feat_dim + book_feat_dim]

                optimizer.zero_grad()

                # 混合精度训练
                with autocast('cuda', enabled=use_amp and self.device.type == 'cuda'):
                    outputs = self.model(features, self.graph_data, user_features, book_features)
                    loss = criterion(outputs, labels)

                # 反向传播和优化
                scaler.scale(loss).backward()

                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += loss.item()

                # 收集预测结果
                all_train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
                all_train_labels.extend(labels.cpu().numpy().flatten())

            # 计算训练集评估指标
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_auc = safe_roc_auc_score(all_train_labels, all_train_preds)
            train_pred_binary = (np.array(all_train_preds) > 0.5).astype(int)
            train_f1 = f1_score(all_train_labels, train_pred_binary, zero_division=0)

            train_losses.append(avg_train_loss)
            train_aucs.append(train_auc)
            train_f1s.append(train_f1)

            # 测试阶段
            test_auc = 0
            test_f1 = 0
            avg_test_loss = 0
            if test_loader:
                self.model.eval()
                epoch_test_loss = 0
                all_test_preds = []
                all_test_labels = []

                with torch.no_grad():
                    for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} - 测试"):
                        features, labels = batch
                        features = features.float().to(self.device, non_blocking=True)
                        labels = labels.float().unsqueeze(1).to(self.device, non_blocking=True)

                        features = torch.nan_to_num(features)
                        labels = torch.nan_to_num(labels)

                        with autocast('cuda', enabled=use_amp and self.device.type == 'cuda'):
                            outputs = self.model(features, self.graph_data,
                                                 features[:, :user_feat_dim],
                                                 features[:, user_feat_dim:user_feat_dim + book_feat_dim])
                            loss = criterion(outputs, labels)

                        epoch_test_loss += loss.item()

                        all_test_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                        all_test_labels.extend(labels.cpu().numpy().flatten())

                avg_test_loss = epoch_test_loss / len(test_loader)
                test_auc = safe_roc_auc_score(all_test_labels, all_test_preds)
                test_pred_binary = (np.array(all_test_preds) > 0.5).astype(int)
                test_f1 = f1_score(all_test_labels, test_pred_binary, zero_division=0)

                test_losses.append(avg_test_loss)
                test_aucs.append(test_auc)
                test_f1s.append(test_f1)

                # 学习率调度
                scheduler.step(test_f1)

                # 早停检查
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_auc = test_auc
                    no_improve = 0
                    torch.save(self.model.state_dict(), "best_model.pth")
                    logging.info(f"  保存最佳模型 (F1: {best_f1:.4f}, AUC: {best_auc:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        early_stop = True

            # 打印训练信息
            log_msg = (f"Epoch {epoch + 1}/{epochs} - "
                       f"训练损失: {avg_train_loss:.4f}, 训练AUC: {train_auc:.4f}, 训练F1: {train_f1:.4f}")
            if test_loader:
                log_msg += (f", 测试损失: {avg_test_loss:.4f}, "
                            f"测试AUC: {test_auc:.4f}, 测试F1: {test_f1:.4f}")
            logging.info(log_msg)

        # 保存最终模型
        torch.save(self.model.state_dict(), "final_model.pth")
        logging.info("最终模型已保存为 final_model.pth")

        # 绘制训练曲线
        self.visualize_training(train_losses, test_losses, train_aucs, test_aucs, train_f1s, test_f1s)

        return {
            'train_loss': train_losses[-1],
            'train_auc': train_aucs[-1],
            'train_f1': train_f1s[-1],
            'test_loss': test_losses[-1] if test_losses else None,
            'test_auc': test_aucs[-1] if test_aucs else None,
            'test_f1': test_f1s[-1] if test_f1s else None
        }

    def visualize_training(self, train_losses, test_losses, train_aucs, test_aucs, train_f1s, test_f1s):
        """可视化训练过程"""
        plt.figure(figsize=(15, 15))

        # 损失曲线
        plt.subplot(3, 1, 1)
        plt.plot(train_losses, label='训练损失')
        if test_losses:
            plt.plot(test_losses, label='测试损失')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # AUC曲线
        plt.subplot(3, 1, 2)
        plt.plot(train_aucs, label='训练AUC')
        if test_aucs:
            plt.plot(test_aucs, label='测试AUC')
        plt.title('AUC曲线')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)

        # F1曲线
        plt.subplot(3, 1, 3)
        plt.plot(train_f1s, label='训练F1')
        if test_f1s:
            plt.plot(test_f1s, label='测试F1')
        plt.title('F1分数曲线')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        logging.info("训练指标曲线已保存为 training_metrics.png")

        # 保存训练指标数据
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'test_loss': test_losses if test_losses else [None] * len(train_losses),
            'train_auc': train_aucs,
            'test_auc': test_aucs if test_aucs else [None] * len(train_losses),
            'train_f1': train_f1s,
            'test_f1': test_f1s if test_f1s else [None] * len(train_losses)
        })
        metrics_df.to_csv('training_metrics.csv', index=False)
        logging.info("训练指标数据已保存为 training_metrics.csv")

    def generate_recommendations(self):
        """生成推荐"""
        logging.info("生成推荐...")

        if self.model is None:
            logging.error("模型未训练，无法生成推荐")
            return pd.DataFrame()

        self.model.eval()
        recommendations = []

        book_feat_dim = 4 + 768
        user_feat_dim = 4

        # 创建预测数据集
        tokenizer, bert_model = self.text_encoder

        predict_dataset = BookRecommendationDataset(
            self.df, tokenizer, bert_model, self.device,
            is_train=False, cache_dir=self.cache_dir
        )

        loader = DataLoader(
            predict_dataset,
            batch_size=1024,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc="  生成推荐"):
                features = batch
                features = features.float().to(self.device, non_blocking=True)
                features = torch.nan_to_num(features)

                outputs = self.model(features, self.graph_data,
                                     features[:, :user_feat_dim],
                                     features[:, user_feat_dim:user_feat_dim + book_feat_dim])
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

                batch_indices = range(len(recommendations), len(recommendations) + len(features))
                user_ids = self.df.iloc[batch_indices]['user_id'].values
                book_ids = self.df.iloc[batch_indices]['book_id'].values

                for user_id, book_id, prob in zip(user_ids, book_ids, probs):
                    recommendations.append({
                        'user_id': user_id,
                        'book_id': book_id,
                        'probability': float(prob)
                    })

        rec_df = pd.DataFrame(recommendations)

        # 为每个用户推荐概率最高的10本书
        top10_recs = rec_df.sort_values(['user_id', 'probability'], ascending=[True, False])
        top10_recs = top10_recs.groupby('user_id').head(10).reset_index(drop=True)

        # 合并豆瓣评分信息
        top10_recs = pd.merge(top10_recs, self.douban_df[['book_id', 'douban_rating']],
                              on='book_id', how='left')

        top10_recs.to_csv('top10_recommendations.csv', index=False)
        logging.info(f"Top10推荐结果已保存到 top10_recommendations.csv，共 {len(top10_recs)} 条推荐")

        # 为每个用户推荐概率最高的书
        top_recs = rec_df.sort_values(['user_id', 'probability'], ascending=[True, False])
        top_recs = top_recs.groupby('user_id').first().reset_index()

        top_recs[['user_id', 'book_id']].to_csv('submission.csv', index=False)
        logging.info(f"单本推荐结果已保存到 submission.csv，共 {len(top_recs)} 条推荐")

        return top10_recs


# 主执行流程
if __name__ == "__main__":
    try:
        # 参数配置
        mode = 'predict'
        epochs = 30
        batch_size = 128
        learning_rate = 0.001
        test_size = 0.2
        patience = 30
        use_amp = True
        sample_size = 1000
        pretrained_model_path = None
        cache_dir = "./cache1"
        bert_model_name = "bert-base-chinese"

        # GNN参数
        gnn_type = 'gat'
        gnn_hidden_dim = 64
        gnn_num_layers = 2
        gnn_heads = 4
        gnn_dropout = 0.2

        # 推荐模型参数
        model_hidden_dim = 64
        model_fusion_layers = 2
        model_predictor_layers = 2
        model_dropout = 0.2

        # 类别不平衡处理
        pos_weight = 2.0

        # 数据路径
        data_path = '/kaggle/input/data11/1data.csv'
        douban_data_path = '/kaggle/input/doubanfe/dr1.csv'

        # 初始化系统
        system = BookRecommendationSystem(
            data_path=data_path,
            douban_data_path=douban_data_path,
            bert_model_name=bert_model_name,
            mode=mode,
            test_size=test_size,
            sample_size=sample_size,
            cache_dir=cache_dir
        )

        # 特征工程
        system.prepare_features()

        # 训练模型
        training_results = system.train_model(
            epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
            patience=patience,
            pretrained_model_path=pretrained_model_path,
            use_amp=use_amp,
            gnn_type=gnn_type,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            gnn_heads=gnn_heads,
            gnn_dropout=gnn_dropout,
            model_hidden_dim=model_hidden_dim,
            model_fusion_layers=model_fusion_layers,
            model_predictor_layers=model_predictor_layers,
            model_dropout=model_dropout,
            pos_weight=pos_weight
        )

        # 输出结果
        logging.info(f"训练完成 - 训练集最终F1分数: {training_results['train_f1']:.4f}")
        if training_results['test_f1'] is not None:
            logging.info(f"训练完成 - 测试集最终F1分数: {training_results['test_f1']:.4f}")

        # 生成推荐
        if mode in ['predict', 'test']:
            top10_recommendations = system.generate_recommendations()
            print("\n=== 推荐结果示例（用户ID、图书ID、推荐概率、豆瓣评分）===")
            if not top10_recommendations.empty:
                print(top10_recommendations[['user_id', 'book_id', 'probability', 'douban_rating']].head(10))
            else:
                print("警告：未生成任何推荐结果，请检查数据或模型配置")
    except Exception as e:
        logging.error(f"程序执行失败，错误原因: {str(e)}")
        import traceback

        traceback.print_exc()