import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import logging
import warnings
import os
import random
import pickle
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

############################################################
# 设置环境变量来抑制警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log1p")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in greater_equal")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


# 安全计算AUC的函数
def safe_roc_auc_score(y_true, y_score):
    """安全计算ROC AUC分数，处理只有一个类别的情况"""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logging.warning(f"只有一个类别存在于标签中，无法计算AUC，返回0.5")
        return 0.5
    return roc_auc_score(y_true, y_score)


class BookFeatureEncoder:
    """使用BERT编码书籍文本特征（修复版）"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        try:
            logging.info("尝试加载BERT模型...")
            # 使用更稳定的加载方式
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', trust_remote_code=True)
            self.model = AutoModel.from_pretrained('bert-base-chinese', trust_remote_code=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            logging.info("BERT模型加载成功")
        except Exception as e:
            logging.error(f"BERT模型加载失败: {str(e)}")
            logging.info("将使用零向量作为文本嵌入")
            # 设置标记，表示使用备用方案
            self.use_fallback = True

    def encode_text(self, text, max_length=64):
        """编码文本为BERT嵌入"""
        if pd.isna(text) or text == "":
            return np.zeros(768)

        # 如果BERT不可用，返回零向量
        if self.tokenizer is None or self.model is None:
            return np.zeros(768)

        try:
            inputs = self.tokenizer(
                str(text),
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding='max_length'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 使用平均池化
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embeddings
        except Exception as e:
            logging.error(f"编码文本时出错: {str(e)}")
            return np.zeros(768)


class BookRecommendationDataset(Dataset):
    """自定义数据集类（修复版）"""

    def __init__(self, df, book_features, feature_encoder, device,
                 is_train=True, cache_dir="./cache"):
        self.df = df.copy()
        self.book_features = book_features
        self.feature_encoder = feature_encoder
        self.device = device
        self.is_train = is_train
        self.cache_dir = cache_dir

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 关键修复：先创建基础特征，再构建图数据
        self._create_basic_features()
        self.graph_data = self._build_interaction_graph()
        self.features, self.labels = self._prepare_features()

        # 检查标签分布
        if self.is_train and self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            logging.info(f"数据集标签分布: {label_distribution}")
            if len(unique) < 2:
                logging.warning("数据集只包含一个类别，可能导致模型训练问题")

    def _create_basic_features(self):
        """创建基础特征用于图构建"""
        # 确保有必要的特征列
        if 'borrow_duration' not in self.df.columns:
            if '还书时间' in self.df.columns and '借阅时间' in self.df.columns:
                duration = (self.df['还书时间'] - self.df['借阅时间']).dt.total_seconds()
                self.df['borrow_duration'] = duration.clip(lower=0) / (24 * 3600)
                self.df['borrow_duration'] = self.df['borrow_duration'].fillna(0)
            else:
                self.df['borrow_duration'] = 0

        if 'renewal_count' not in self.df.columns:
            self.df['renewal_count'] = self.df.get('续借次数', 0).fillna(0).astype(float)

        if 'is_renewed' not in self.df.columns:
            self.df['is_renewed'] = self.df.get('是否续借', 0).fillna(0).astype(int)

    def _build_interaction_graph(self):
        """构建用户-图书交互图"""
        logging.info("  构建用户-图书交互图...")

        user_ids = self.df['user_id'].unique()
        book_ids = self.df['book_id'].unique()

        user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        book_mapping = {book_id: idx + len(user_ids) for idx, book_id in enumerate(book_ids)}

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

        return graph_data

    def _prepare_features(self):
        """准备特征 - 修复版本"""
        logging.info("  准备特征...")

        # 用户特征
        user_ids = self.df['user_id'].unique()
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

        user_groups = self.df.groupby('user_id')
        user_features = user_groups.agg(
            avg_duration=('borrow_duration', 'mean'),
            total_borrows=('book_id', 'count'),
            renewal_rate=('is_renewed', 'mean')
        ).reset_index(drop=True)
        user_features = user_features.fillna(0).values

        # 图书特征 - 使用预计算的特征
        book_ids = self.df['book_id'].unique()

        # 生成缓存文件名
        import hashlib
        book_ids_str = "_".join(map(str, sorted(book_ids)))
        hash_obj = hashlib.md5(book_ids_str.encode())
        cache_filename = f"book_features_{hash_obj.hexdigest()}.pkl"
        self.cache_path = os.path.join(self.cache_dir, cache_filename)

        # 检查缓存是否存在
        if os.path.exists(self.cache_path):
            logging.info(f"  加载预计算的图书特征缓存: {self.cache_path}")
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

                # 获取图书的文本特征
                summary = ""
                reading_style = ""
                if book_id in self.book_features.index:
                    book_info = self.book_features.loc[book_id]
                    summary = book_info.get('summary', '')
                    reading_style = book_info.get('reading_style', '')

                # 使用BERT编码文本
                summary_embedding = self.feature_encoder.encode_text(summary)
                reading_style_embedding = self.feature_encoder.encode_text(reading_style, max_length=32)

                book_id_to_features[book_id] = [
                    avg_duration, total_borrows, renewal_rate,
                    *summary_embedding,
                    *reading_style_embedding
                ]

            # 保存特征缓存
            logging.info(f"  保存图书特征缓存到: {self.cache_path}")
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

            # 交互特征
            duration = row['borrow_duration']
            renewal = row['is_renewed']
            renewal_count = row['renewal_count']
            last_borrow_days = row.get('days_since_last_borrow', 0)

            feature_vec = np.concatenate([
                user_feat,
                book_feat,
                [duration, renewal, renewal_count, last_borrow_days]
            ])
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
    """图神经网络模型"""

    def __init__(self, num_user_features, num_book_features, hidden_dim=128, heads=1):
        super().__init__()

        self.user_embedding = nn.Linear(num_user_features, hidden_dim)
        self.book_embedding = nn.Linear(num_book_features, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim // 2, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim // 2 * heads, hidden_dim, dropout=0.2)

        self.bn1 = nn.BatchNorm1d(hidden_dim // 2 * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph_data, user_features, book_features):
        user_emb = F.relu(self.user_embedding(user_features))
        book_emb = F.relu(self.book_embedding(book_features))
        x = torch.cat([user_emb, book_emb], dim=0)

        # 检查节点数量匹配
        if x.size(0) != graph_data.num_nodes:
            if x.size(0) < graph_data.num_nodes:
                padding = torch.zeros(graph_data.num_nodes - x.size(0), x.size(1)).to(x.device)
                x = torch.cat([x, padding], dim=0)
            else:
                x = x[:graph_data.num_nodes, :]

        # 检查边索引有效性
        if graph_data.edge_index.numel() > 0:
            max_idx = graph_data.edge_index.max().item()
            if max_idx >= x.size(0):
                mask = (graph_data.edge_index[0] < x.size(0)) & (graph_data.edge_index[1] < x.size(0))
                graph_data.edge_index = graph_data.edge_index[:, mask]

        x = self.conv1(x, graph_data.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, graph_data.edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class BookRecommendationModel(nn.Module):
    """完整的推荐系统模型"""

    def __init__(self, num_features, graph_model, hidden_dim=128):
        super().__init__()
        self.graph_model = graph_model
        self.hidden_dim = hidden_dim

        self.temporal_module = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        fusion_input_dim = num_features - 4 + (hidden_dim // 2) + hidden_dim * 2

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, features, graph_data, user_features, book_features):
        temporal_features = features[:, -4:]
        temporal_emb = self.temporal_module(temporal_features)

        graph_emb = self.graph_model(graph_data, user_features, book_features)

        user_graph_emb = graph_emb[:graph_data.num_users]
        book_graph_emb = graph_emb[graph_data.num_users:]

        batch_size = features.size(0)

        if len(user_graph_emb) == 0:
            user_graph_emb = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        elif user_graph_emb.size(0) < batch_size:
            repeat_factor = (batch_size // user_graph_emb.size(0)) + 1
            user_graph_emb = user_graph_emb.repeat(repeat_factor, 1)[:batch_size]
        else:
            user_graph_emb = user_graph_emb[:batch_size]

        if len(book_graph_emb) == 0:
            book_graph_emb = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        elif book_graph_emb.size(0) < batch_size:
            repeat_factor = (batch_size // book_graph_emb.size(0)) + 1
            book_graph_emb = book_graph_emb.repeat(repeat_factor, 1)[:batch_size]
        else:
            book_graph_emb = book_graph_emb[:batch_size]

        combined = torch.cat([
            features[:, :-4],
            temporal_emb,
            user_graph_emb,
            book_graph_emb
        ], dim=1)

        fused = self.fusion_layer(combined)
        return self.predictor(fused)


class BookRecommendationSystem:
    def __init__(self, data_path, book_features_path, mode='train',
                 test_size=0.2, sample_size=None, cache_dir="./cache"):
        self.data_path = data_path
        self.book_features_path = book_features_path
        self.mode = mode
        self.test_size = test_size
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        self.df = None
        self.book_features = None
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.graph_data = None
        self.model = None
        self.gnn_model = None
        self.feature_encoder = None
        self.scaler = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用 {'GPU' if torch.cuda.is_available() else 'CPU'} 进行计算")

        self.load_data()

        # 初始化特征编码器
        self.feature_encoder = BookFeatureEncoder()

    def load_data(self):
        """加载并预处理数据"""
        logging.info("加载数据...")
        try:
            self.df = pd.read_csv(self.data_path, engine='python')
            logging.info(f"主数据加载成功，共 {len(self.df)} 条记录")

            self.book_features = pd.read_csv(self.book_features_path, engine='python')
            self.book_features.set_index('book_id', inplace=True)
            logging.info(f"书籍特征数据加载成功，共 {len(self.book_features)} 条记录")
        except Exception as e:
            logging.error(f"加载数据失败: {str(e)}")
            raise

        required_columns = ['user_id', 'book_id', '借阅时间']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        keep_columns = required_columns + ['还书时间', '续借时间', '续借次数', '是否续借', '一级分类']
        keep_columns = [col for col in keep_columns if col in self.df.columns]
        self.df = self.df[keep_columns].dropna(subset=required_columns)

        if self.mode == 'test' and self.sample_size is not None and len(self.df) > self.sample_size:
            if 'will_borrow_again' in self.df.columns:
                self.df = self.df.groupby('will_borrow_again', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), self.sample_size // 2))
                )
            else:
                self.df = self.df.sample(self.sample_size, random_state=42)
            logging.info(f"测试模式：使用 {len(self.df)} 条样本数据")

        date_cols = ['借阅时间', '还书时间', '续借时间']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce', format='%Y-%m-%d %H:%M:%S')
                except:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        logging.info(f"数据集加载完成，共 {len(self.df)} 条记录")

    def prepare_features(self):
        """准备特征工程"""
        logging.info("开始特征工程...")

        # 1. 创建目标变量
        logging.info("步骤1/5: 创建目标变量...")
        self._create_target_variable()

        if 'will_borrow_again' in self.df.columns:
            target_dist = self.df['will_borrow_again'].value_counts(normalize=True)
            logging.info(f"目标变量分布: {target_dist.to_dict()}")
            if len(target_dist) < 2:
                logging.warning("目标变量只包含一个类别，这会导致模型训练问题")

        # 2. 创建基础特征
        logging.info("步骤2/5: 创建基础特征...")
        self._create_basic_features()

        # 3. 类别特征编码
        logging.info("步骤3/5: 类别特征编码...")
        self._encode_categorical_features()

        # 4. 数值特征处理
        logging.info("步骤4/5: 数值特征处理...")
        self._process_numerical_features()

        # 5. 创建数据集
        logging.info("步骤5/5: 创建数据集...")
        self._create_dataset()

        logging.info("特征工程完成")

    def _create_target_variable(self):
        """创建目标变量：是否会二次借阅"""
        borrow_counts = self.df.groupby(['user_id', 'book_id']).size().reset_index(name='borrow_count')
        self.df = pd.merge(self.df, borrow_counts, on=['user_id', 'book_id'], how='left')

        self.df['is_first_borrow'] = self.df.groupby(['user_id', 'book_id'])['借阅时间'].rank(method='first') == 1

        self.df['will_borrow_again'] = 0
        self.df.loc[(self.df['is_first_borrow']) & (self.df['borrow_count'] > 1), 'will_borrow_again'] = 1

        self.df = self.df[self.df['is_first_borrow']].copy()

    def _create_basic_features(self):
        """创建基础特征"""
        if '还书时间' in self.df.columns and '借阅时间' in self.df.columns:
            duration = (self.df['还书时间'] - self.df['借阅时间']).dt.total_seconds()
            self.df['borrow_duration'] = duration.clip(lower=0) / (24 * 3600)
            self.df['borrow_duration'] = self.df['borrow_duration'].fillna(0)

        self.df['renewal_count'] = self.df.get('续借次数', 0).fillna(0).astype(float)
        self.df['is_renewed'] = self.df.get('是否续借', 0).fillna(0).astype(int)

        self.df['user_borrow_count'] = self.df.groupby('user_id')['book_id'].transform('count')
        self.df['user_avg_duration'] = self.df.groupby('user_id')['borrow_duration'].transform('mean')
        self.df['user_renewal_rate'] = self.df.groupby('user_id')['is_renewed'].transform('mean')

        self.df['book_borrow_count'] = self.df.groupby('book_id')['user_id'].transform('count')
        self.df['book_avg_duration'] = self.df.groupby('book_id')['borrow_duration'].transform('mean')
        self.df['book_renewal_rate'] = self.df.groupby('book_id')['is_renewed'].transform('mean')

        self.df = self.df.sort_values(['user_id', '借阅时间'])
        self.df['days_since_last_borrow'] = self.df.groupby('user_id')['借阅时间'].diff().dt.days.fillna(0).clip(
            lower=0)

    def _encode_categorical_features(self):
        """编码类别特征"""
        if '一级分类' in self.df.columns:
            category_counts = self.df['一级分类'].value_counts()
            self.df['category_encoded'] = self.df['一级分类'].map(category_counts)
            self.cat_features = ['category_encoded']
        else:
            self.cat_features = []
            logging.warning("没有找到类别特征列，跳过类别编码")

    def _process_numerical_features(self):
        """标准化数值特征"""
        num_features = [
                           'borrow_duration', 'renewal_count', 'is_renewed',
                           'user_borrow_count', 'user_avg_duration', 'user_renewal_rate',
                           'book_borrow_count', 'book_avg_duration', 'book_renewal_rate',
                           'days_since_last_borrow'
                       ] + self.cat_features

        self.num_features = [col for col in num_features if col in self.df.columns]

        if not self.num_features:
            logging.warning("没有数值特征可用于标准化")
            return

        for col in self.num_features:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[self.num_features])
        for i, col in enumerate(self.num_features):
            self.df[col] = scaled_features[:, i]

    def _create_dataset(self):
        """创建数据集"""
        self.full_dataset = BookRecommendationDataset(
            self.df, self.book_features, self.feature_encoder, self.device,
            is_train=True, cache_dir=self.cache_dir
        )

        if self.mode == 'train':
            from sklearn.model_selection import train_test_split
            from torch.utils.data import Subset

            labels = self.df['will_borrow_again'].values
            train_indices, test_indices = train_test_split(
                range(len(self.full_dataset)),
                test_size=self.test_size,
                random_state=42,
                stratify=labels
            )

            self.train_dataset = Subset(self.full_dataset, train_indices)
            self.test_dataset = Subset(self.full_dataset, test_indices)

            logging.info(f"数据集划分: 训练集 {len(self.train_dataset)} 条, 测试集 {len(self.test_dataset)} 条")
        elif self.mode == 'predict':
            self.train_dataset = self.full_dataset
            logging.info(f"预测模式: 使用全部 {len(self.train_dataset)} 条数据训练")
        elif self.mode == 'test':
            self.train_dataset = self.full_dataset
            logging.info(f"测试模式: 使用 {len(self.train_dataset)} 条数据训练")

        self.graph_data = self.full_dataset.get_graph_data()

    def train_model(self, epochs=20, lr=0.001, batch_size=512, patience=5, use_amp=True):
        """训练模型并评估"""
        logging.info("开始训练模型...")

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
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        ) if self.test_dataset else None

        num_features = self.full_dataset.features.shape[1]
        logging.info(f"输入特征维度: {num_features}")

        # 创建模型
        self.gnn_model = GNNModel(
            num_user_features=3,
            num_book_features=1539
        ).to(self.device)

        self.model = BookRecommendationModel(
            num_features=num_features,
            graph_model=self.gnn_model,
            hidden_dim=128
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )

        criterion = nn.BCEWithLogitsLoss()

        scaler = GradScaler('cuda', enabled=use_amp and self.device.type == 'cuda')

        train_losses = []
        test_losses = []
        train_aucs = []
        test_aucs = []
        best_auc = 0
        no_improve = 0
        early_stop = False

        for epoch in range(epochs):
            if early_stop:
                logging.info(f"早停: 在 {epoch} 轮停止训练")
                break

            self.model.train()
            epoch_train_loss = 0
            all_train_preds = []
            all_train_labels = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - 训练"):
                features, labels = batch
                features = features.float().to(self.device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(self.device, non_blocking=True)

                features = torch.nan_to_num(features)
                labels = torch.nan_to_num(labels)

                user_features = features[:, :3]
                book_features = features[:, 3:1542]

                optimizer.zero_grad()

                with autocast('cuda', enabled=use_amp and self.device.type == 'cuda'):
                    outputs = self.model(features, self.graph_data, user_features, book_features)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += loss.item()

                all_train_preds.extend(torch.sigmoid(outputs).detach().cpu().float().numpy().flatten())
                all_train_labels.extend(labels.cpu().numpy().flatten())

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_auc = safe_roc_auc_score(all_train_labels, all_train_preds)
            train_losses.append(avg_train_loss)
            train_aucs.append(train_auc)

            test_auc = 0
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

                        with autocast('cuda', enabled=use_amp and self.device.type == 'cuda'):
                            outputs = self.model(features, self.graph_data,
                                                 features[:, :3], features[:, 3:1542])
                            loss = criterion(outputs, labels)

                        epoch_test_loss += loss.item()

                        all_test_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                        all_test_labels.extend(labels.cpu().numpy().flatten())

                avg_test_loss = epoch_test_loss / len(test_loader)
                test_auc = safe_roc_auc_score(all_test_labels, all_test_preds)
                test_losses.append(avg_test_loss)
                test_aucs.append(test_auc)

                scheduler.step(test_auc)

                if test_auc > best_auc:
                    best_auc = test_auc
                    no_improve = 0
                    torch.save(self.model.state_dict(), "best_model.pth")
                    logging.info(f"  保存最佳模型 (AUC: {best_auc:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        early_stop = True

            log_msg = f"Epoch {epoch + 1}/{epochs} - 训练损失: {avg_train_loss:.4f}, 训练AUC: {train_auc:.4f}"
            if test_loader:
                log_msg += f", 测试损失: {avg_test_loss:.4f}, 测试AUC: {test_auc:.4f}"
            logging.info(log_msg)

        torch.save(self.model.state_dict(), "final_model.pth")
        logging.info("最终模型已保存为 final_model.pth")

        self.visualize_training(train_losses, test_losses, train_aucs, test_aucs)

        return train_losses[-1]

    def visualize_training(self, train_losses, test_losses, train_aucs, test_aucs):
        """可视化训练过程"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='训练损失')
        if test_losses:
            plt.plot(test_losses, label='测试损失')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(train_aucs, label='训练AUC')
        if test_aucs:
            plt.plot(test_aucs, label='测试AUC')
        plt.title('AUC曲线')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        logging.info("训练指标曲线已保存为 training_metrics.png")

        metrics_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'test_loss': test_losses if test_losses else [None] * len(train_losses),
            'train_auc': train_aucs,
            'test_auc': test_aucs if test_aucs else [None] * len(train_losses)
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

        predict_dataset = BookRecommendationDataset(
            self.df, self.book_features, self.feature_encoder, self.device,
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

                outputs = self.model(features, self.graph_data,
                                     features[:, :3], features[:, 3:1542])
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

        top_recs = rec_df.sort_values(['user_id', 'probability'], ascending=[True, False])
        top_recs = top_recs.groupby('user_id').head(1).reset_index(drop=True)

        top_recs = top_recs[['user_id', 'book_id']]

        output_file = 'user_book_recommendations.csv'
        top_recs.to_csv(output_file, index=False)
        logging.info(f"推荐结果已保存到 {output_file}，共 {len(top_recs)} 条推荐")

        return top_recs


# 主执行流程
if __name__ == "__main__":
    try:
        # 参数设置
        MODE = 'predict'
        DATA_PATH = '/kaggle/input/ashhad/1data.csv'
        BOOK_FEATURES_PATH = '/kaggle/input/damoxing/books_with_features.csv'
        CACHE_DIR = "./book-features-cache"

        EPOCHS = 30
        BATCH_SIZE = 128
        LEARNING_RATE = 0.001
        TEST_SIZE = 0.2
        PATIENCE = 5
        USE_AMP = True
        SAMPLE_SIZE = 1000

        if MODE == 'train':
            SAMPLE_SIZE = None
        elif MODE == 'predict':
            TEST_SIZE = 0.0
            SAMPLE_SIZE = None
            PATIENCE = None
        elif MODE == 'test':
            EPOCHS = 5
            TEST_SIZE = 0.2
            PATIENCE = None

        # 初始化系统
        system = BookRecommendationSystem(
            data_path=DATA_PATH,
            book_features_path=BOOK_FEATURES_PATH,
            mode=MODE,
            test_size=TEST_SIZE,
            sample_size=SAMPLE_SIZE,
            cache_dir=CACHE_DIR
        )

        # 准备特征
        system.prepare_features()

        # 训练模型
        system.train_model(
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            use_amp=USE_AMP
        )

        # 生成推荐结果
        if MODE == 'predict' or MODE == 'test':
            recommendations = system.generate_recommendations()
            print("\n推荐结果示例:")
            if not recommendations.empty:
                print(recommendations.head(10))
            else:
                print("未生成任何推荐")

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        import traceback

        traceback.print_exc()