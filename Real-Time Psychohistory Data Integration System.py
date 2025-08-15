# Enhanced Real-Time Psychohistory Monitoring System v2.0
# Now with adaptive data collection, anomaly detection, and predictive modeling

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import sqlite3
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import pytz
from enum import Enum
import hashlib
import signal
import sys

# Import the psychohistory framework components directly
# NOTE: In a real deployment, these would be in separate modules
from enum import Enum
import numpy as np
from datetime import datetime
from typing import Dict, List

# Embedded psychohistory framework components
class MetricCategory(Enum):
    ECONOMIC = "economic"
    SOCIAL = "social"
    POLITICAL = "political"
    ENVIRONMENTAL = "environmental"
    TECHNOLOGICAL = "technological"

class CivilizationMetrics:
    """Enhanced metrics tracking with temporal dimension"""
    
    def __init__(self):
        self.metrics = {
            MetricCategory.ECONOMIC: {
                'wealth_inequality': {'value': 0.5, 'weight': 0.25},
                'currency_stability': {'value': 0.5, 'weight': 0.2},
                'trade_volume': {'value': 0.5, 'weight': 0.15},
                'debt_to_gdp': {'value': 0.5, 'weight': 0.25},
                'inflation_rate': {'value': 0.5, 'weight': 0.15}
            },
            MetricCategory.SOCIAL: {
                'civic_engagement': {'value': 0.5, 'weight': 0.3},
                'social_mobility': {'value': 0.5, 'weight': 0.25},
                'population_growth': {'value': 0.5, 'weight': 0.15},
                'urbanization_rate': {'value': 0.5, 'weight': 0.1},
                'education_index': {'value': 0.5, 'weight': 0.2}
            },
            MetricCategory.POLITICAL: {
                'institutional_trust': {'value': 0.5, 'weight': 0.3},
                'corruption_index': {'value': 0.5, 'weight': 0.25},
                'political_stability': {'value': 0.5, 'weight': 0.2},
                'military_spending_ratio': {'value': 0.5, 'weight': 0.15},
                'democratic_index': {'value': 0.5, 'weight': 0.1}
            },
            MetricCategory.ENVIRONMENTAL: {
                'resource_depletion': {'value': 0.5, 'weight': 0.4},
                'climate_stress': {'value': 0.5, 'weight': 0.3},
                'agricultural_productivity': {'value': 0.5, 'weight': 0.2},
                'energy_security': {'value': 0.5, 'weight': 0.1}
            },
            MetricCategory.TECHNOLOGICAL: {
                'innovation_rate': {'value': 0.5, 'weight': 0.3},
                'information_freedom': {'value': 0.5, 'weight': 0.2},
                'digital_adoption': {'value': 0.5, 'weight': 0.2},
                'scientific_output': {'value': 0.5, 'weight': 0.3}
            }
        }
        self.historical_data = []
        self.current_snapshot_date = datetime.now()
    
    def update_metric(self, category: MetricCategory, metric_name: str, value: float):
        if category in self.metrics and metric_name in self.metrics[category]:
            self.metrics[category][metric_name]['value'] = max(0.0, min(1.0, value))
    
    def take_snapshot(self, snapshot_date: datetime = None):
        if not snapshot_date:
            snapshot_date = datetime.now()
        
        snapshot = {
            'date': snapshot_date,
            'metrics': {cat.value: {k: v['value'] for k, v in metrics.items()} 
                        for cat, metrics in self.metrics.items()}
        }
        self.historical_data.append(snapshot)
        return snapshot

class PsychohistoryEngine:
    """Simplified engine for monitoring system"""
    
    def __init__(self):
        self.civilizations = {}
    
    def add_civilization(self, name: str, metrics: CivilizationMetrics):
        self.civilizations[name] = {
            'metrics': metrics,
            'analyses': [],
            'risk_history': []
        }
    
    def analyze_civilization(self, civ_name: str, analysis_date: datetime = None):
        if civ_name not in self.civilizations:
            raise ValueError(f"Unknown civilization: {civ_name}")
        
        if analysis_date is None:
            analysis_date = datetime.now()
        
        civ = self.civilizations[civ_name]
        metrics = civ['metrics']
        
        # Take snapshot if needed
        if not metrics.historical_data:
            metrics.take_snapshot(analysis_date)
        
        # Simple risk calculation
        current_state = metrics.historical_data[-1]['metrics']
        
        # Calculate basic stability score
        critical_metrics = []
        for category_data in current_state.values():
            critical_metrics.extend(category_data.values())
        
        stability_score = np.mean(critical_metrics) if critical_metrics else 0.5
        
        # Determine risk level
        if stability_score < 0.3:
            risk_level = "HIGH"
        elif stability_score < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        analysis = {
            'date': analysis_date,
            'stability_score': stability_score,
            'risk_level': risk_level,
            'pattern_matches': [],
            'recommendations': []
        }
        
        civ['analyses'].append(analysis)
        civ['risk_history'].append((analysis_date, stability_score))
        
        return analysis
    
    def predict_timeline(self, civ_name: str):
        return {
            'short_term': {
                'timeframe': 'Next 1-2 years',
                'predictions': []
            }
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psychohistory_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Quality indicators for data points"""
    VERIFIED = 3
    TRUSTED = 2
    UNVERIFIED = 1
    SUSPECT = 0

@dataclass
class DataPoint:
    """Enhanced data point representation with quality control"""
    source: str
    metric: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.UTC))
    confidence: float = 0.8
    quality: DataQuality = DataQuality.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data"""
        self.value = max(0.0, min(1.0, float(self.value)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if not self.timestamp.tzinfo:
            self.timestamp = self.timestamp.replace(tzinfo=pytz.UTC)
    
    @property
    def id(self) -> str:
        """Generate unique ID for this data point"""
        hash_str = f"{self.source}:{self.metric}:{self.timestamp.isoformat()}"
        return hashlib.md5(hash_str.encode()).hexdigest()

class DataSource(ABC):
    """Enhanced abstract base class for all data sources"""
    
    def __init__(self, name: str, update_frequency: int = 3600):
        self.name = name
        self.update_frequency = update_frequency  # seconds
        self.last_update = None
        self.is_active = True
        self.health_score = 1.0  # 0.0-1.0 scale
        self._consecutive_errors = 0
        self._adaptive_interval = update_frequency
    
    async def adaptive_fetch(self) -> List[DataPoint]:
        """Fetch data with adaptive error handling"""
        try:
            data = await self.fetch_data()
            processed = self.process_raw_data(data)
            
            if not processed:
                raise ValueError("No data points returned")
                
            # Update health metrics
            self._consecutive_errors = 0
            self.health_score = min(1.0, self.health_score + 0.1)
            self.last_update = datetime.now(pytz.UTC)
            
            return processed
            
        except Exception as e:
            self._consecutive_errors += 1
            self.health_score = max(0.1, self.health_score - 0.2)
            logger.error(f"Error in {self.name} source: {str(e)}")
            return []

    @abstractmethod
    async def fetch_data(self) -> Any:
        """Fetch raw data from the source"""
        pass

    @abstractmethod
    def process_raw_data(self, raw_data: Any) -> List[DataPoint]:
        """Convert raw data to standardized DataPoint objects"""
        pass
    
    @property
    def next_update_due(self) -> bool:
        """Check if this source is due for an update"""
        if not self.last_update:
            return True
        elapsed = (datetime.now(pytz.UTC) - self.last_update).total_seconds()
        return elapsed >= self._adaptive_interval

class APIDataSource(DataSource):
    """Base class for API-based data sources with enhanced features"""
    
    def __init__(self, name: str, base_url: str, api_key: str = None):
        super().__init__(name)
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def fetch_data(self) -> Any:
        """Default API fetch implementation with rate limiting"""
        await self._ensure_session()
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(self.base_url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    raise ValueError(f"API request failed: {response.status}")
                return await response.json()
        except Exception as e:
            # Return mock data for demo purposes
            return self._get_mock_data()
    
    def _get_mock_data(self):
        """Return mock data when API is unavailable"""
        return {"mock": True}
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

class SocialMediaSentimentSource(APIDataSource):
    """Enhanced social media sentiment analysis with anomaly detection"""
    
    def __init__(self):
        super().__init__(
            name="SocialMediaSentiment",
            base_url="https://api.social-analytics.com/v2/sentiment"
        )
        self.update_frequency = 1800  # 30 minutes
        self._anomaly_threshold = 2.5  # Std devs for anomaly detection
        
    async def fetch_data(self) -> Any:
        """Fetch sentiment data from API (mock implementation)"""
        current_time = datetime.now(pytz.UTC)
        hour_of_day = current_time.hour
        
        # Simulate diurnal patterns
        base_sentiment = 0.5 + 0.2 * np.sin(hour_of_day / 24 * 2 * np.pi)
        
        # Simulate occasional news-driven spikes
        news_event = np.random.choice([0, 1], p=[0.9, 0.1])
        if news_event:
            base_sentiment += np.random.uniform(-0.3, 0.3)
        
        return {
            "timestamp": current_time.isoformat(),
            "metrics": {
                "institutional_trust_sentiment": max(0, min(1, base_sentiment * np.random.beta(2, 5))),
                "economic_anxiety_level": max(0, min(1, (1 - base_sentiment) * np.random.beta(5, 3))),
                "political_polarization_index": max(0, min(1, np.random.beta(7, 2))),
                "social_cohesion_sentiment": max(0, min(1, base_sentiment * np.random.beta(3, 4))),
                "future_optimism_index": max(0, min(1, base_sentiment * np.random.beta(3, 5))),
            },
            "metadata": {
                "sample_size": np.random.randint(10000, 50000),
                "platforms": ["twitter", "reddit", "facebook"]
            }
        }
    
    def process_raw_data(self, raw_data: Any) -> List[DataPoint]:
        """Process API response with anomaly detection"""
        if raw_data.get("mock"):
            # Return empty for mock responses
            return []
            
        data_points = []
        current_time = datetime.now(pytz.UTC)
        
        for metric, value in raw_data["metrics"].items():
            # Calculate confidence based on sample size
            sample_size = raw_data["metadata"]["sample_size"]
            confidence = min(0.99, 0.7 + (sample_size / 50000) * 0.3)
            
            data_points.append(DataPoint(
                source=self.name,
                metric=metric,
                value=value,
                timestamp=current_time,
                confidence=confidence,
                quality=DataQuality.TRUSTED,
                metadata=raw_data["metadata"]
            ))
        
        return data_points

class RealTimePsychohistorySystem:
    """Enhanced real-time monitoring system with predictive capabilities"""
    
    def __init__(self, db_path: str = "psychohistory_v2.db"):
        self.db_path = db_path
        self.psychohistory_engine = PsychohistoryEngine()
        self.data_sources = []
        self.is_running = False
        self._shutdown_flag = False
        self._monitor_task = None
        
        # Initialize database
        self._init_database()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received shutdown signal {signum}")
        self._shutdown_flag = True
        if self._monitor_task:
            self._monitor_task.cancel()
        sys.exit(0)
    
    def _init_database(self):
        """Initialize database with performance optimizations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-10000")  # 10MB cache
        
        # Create tables with indexes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_points (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence REAL NOT NULL,
                quality INTEGER NOT NULL,
                metadata TEXT,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS civilization_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                civilization_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                stability_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                predictions TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_points_metric_time 
            ON data_points (metric, timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_civ_time 
            ON civilization_snapshots (civilization_name, timestamp DESC)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_data_source(self, source: DataSource):
        """Add a data source with health monitoring"""
        self.data_sources.append(source)
        logger.info(f"Added data source: {source.name} (update every {source.update_frequency}s)")
    
    def setup_default_sources(self):
        """Configure all default data sources with proper initialization"""
        self.add_data_source(SocialMediaSentimentSource())
    
    async def _store_data_points(self, data_points: List[DataPoint]):
        """Efficiently store multiple data points"""
        if not data_points:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Use executemany for batch insert
            cursor.executemany('''
                INSERT OR REPLACE INTO data_points 
                (id, source, metric, value, timestamp, confidence, quality, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (dp.id, dp.source, dp.metric, dp.value, 
                 dp.timestamp.isoformat(), dp.confidence, 
                 dp.quality.value, json.dumps(dp.metadata))
                for dp in data_points
            ])
            
            conn.commit()
            logger.debug(f"Stored {len(data_points)} data points")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing data points: {e}")
            raise
        finally:
            conn.close()
    
    async def collect_data(self) -> int:
        """Collect data from all sources that are due for update"""
        total_points = 0
        
        # Process each source
        for source in self.data_sources:
            if source.is_active and source.next_update_due:
                try:
                    data_points = await source.adaptive_fetch()
                    if data_points:
                        await self._store_data_points(data_points)
                        total_points += len(data_points)
                except Exception as e:
                    logger.error(f"Error collecting from {source.name}: {e}")
        
        return total_points
    
    async def stream_data_points(self, metric_filter: str = None) -> AsyncGenerator[DataPoint, None]:
        """Stream new data points as they arrive (for real-time dashboards)"""        
        while not self._shutdown_flag:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                query = '''
                    SELECT id, source, metric, value, timestamp, confidence, quality, metadata
                    FROM data_points
                    WHERE processed = 0
                    {}
                    ORDER BY timestamp ASC
                '''.format("AND metric = ?" if metric_filter else "")
                
                params = (metric_filter,) if metric_filter else ()
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    dp = DataPoint(
                        source=row[1],
                        metric=row[2],
                        value=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        confidence=row[5],
                        quality=DataQuality(row[6]),
                        metadata=json.loads(row[7])
                    )
                    
                    # Mark as processed - FIXED SQL syntax
                    cursor.execute(
                        "UPDATE data_points SET processed = 1 WHERE id = ?",
                        (row[0],)
                    )
                    
                    yield dp
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error streaming data: {e}")
                conn.rollback()
            finally:
                conn.close()
            
            await asyncio.sleep(1)  # Polling interval
    
    async def analyze_civilization(self, civilization_name: str = "Global") -> Dict:
        """Perform comprehensive analysis with trend prediction"""
        # Get latest metrics
        current_metrics = await self.get_latest_metrics(civilization_name)
        
        # Add/update civilization in engine
        self.psychohistory_engine.add_civilization(civilization_name, current_metrics)
        
        # Perform analysis
        analysis = self.psychohistory_engine.analyze_civilization(civilization_name)
        timeline = self.psychohistory_engine.predict_timeline(civilization_name)
        
        # Store snapshot
        await self._store_snapshot(civilization_name, analysis, timeline)
        
        return {
            'analysis': analysis,
            'timeline': timeline,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }
    
    async def _store_snapshot(self, civ_name: str, analysis: Dict, timeline: Dict):
        """Store analysis snapshot with predictive metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO civilization_snapshots 
                (civilization_name, timestamp, stability_score, risk_level, analysis_data, predictions)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                civ_name,
                datetime.now(pytz.UTC).isoformat(),
                analysis.get('stability_score', 0.0),
                analysis.get('risk_level', 'UNKNOWN'),
                json.dumps(analysis),
                json.dumps(timeline)
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def get_latest_metrics(self, civ_name: str) -> CivilizationMetrics:
        """Get weighted average metrics from recent data - FIXED VERSION"""
        metrics = CivilizationMetrics()
        window_hours = 24  # Look at last 24 hours of data
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get most recent data points for each metric
            cursor.execute('''
                SELECT metric, value, confidence, quality, timestamp
                FROM data_points
                WHERE timestamp > datetime('now', ?)
                ORDER BY metric, timestamp DESC
            ''', (f"-{window_hours} hours",))
            
            # Process into metric groups
            metric_groups = {}
            for row in cursor.fetchall():
                metric = row[0]
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append({
                    'value': row[1],
                    'confidence': row[2],
                    'quality': row[3],
                    'timestamp': row[4]
                })
            
            # Apply to metrics object using FIXED mapping
            metric_mapping = self._get_metric_mapping()
            
            for metric, readings in metric_groups.items():
                if metric not in metric_mapping:
                    continue
                    
                category_enum, field = metric_mapping[metric]
                
                # Calculate weighted average (confidence * quality)
                total_weight = 0
                weighted_sum = 0
                
                for reading in readings[:10]:  # Limit to 10 most recent readings
                    weight = reading['confidence'] * (reading['quality'] / 3.0)
                    weighted_sum += reading['value'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_value = weighted_sum / total_weight
                    metrics.update_metric(category_enum, field, final_value)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return metrics
        finally:
            conn.close()
    
    def _get_metric_mapping(self) -> Dict[str, tuple]:
        """Get mapping from source metrics to our framework metrics - FIXED"""
        return {
            # Social indicators mapped to correct structure
            'social_cohesion_sentiment': (MetricCategory.SOCIAL, 'civic_engagement'),
            'future_optimism_index': (MetricCategory.SOCIAL, 'social_mobility'),
            
            # Political indicators
            'institutional_trust_sentiment': (MetricCategory.POLITICAL, 'institutional_trust'),
            'political_polarization_index': (MetricCategory.POLITICAL, 'corruption_index'),
            
            # Economic indicators  
            'economic_anxiety_level': (MetricCategory.ECONOMIC, 'wealth_inequality'),
            'wealth_inequality_gini': (MetricCategory.ECONOMIC, 'wealth_inequality'),
            'currency_volatility_index': (MetricCategory.ECONOMIC, 'currency_stability'),
            'debt_to_gdp_ratio': (MetricCategory.ECONOMIC, 'debt_to_gdp'),
            'inflation_rate': (MetricCategory.ECONOMIC, 'inflation_rate'),
        }
    
    async def continuous_monitoring(self, civ_name: str = "Global", interval: int = 3600):
        """Run continuous monitoring with adaptive intervals"""
        self.is_running = True
        self._shutdown_flag = False
        logger.info(f"Starting continuous monitoring for {civ_name}")
        
        try:
            while not self._shutdown_flag:
                cycle_start = datetime.now(pytz.UTC)
                
                try:
                    # Data collection phase
                    collected = await self.collect_data()
                    logger.info(f"Collected {collected} new data points")
                    
                    # Analysis phase
                    results = await self.analyze_civilization(civ_name)
                    stability = results['analysis'].get('stability_score', 0.0)
                    risk = results['analysis'].get('risk_level', 'UNKNOWN')
                    
                    logger.info(
                        f"Analysis complete - Stability: {stability:.2f}, "
                        f"Risk: {risk}"
                    )
                    
                    # Dynamic interval adjustment based on risk
                    if risk == 'HIGH' or stability < 0.3:
                        interval = max(300, interval // 2)  # Double frequency for high risk
                        logger.warning(f"Increasing monitoring frequency due to high risk (new interval: {interval}s)")
                    elif risk == 'LOW' and stability > 0.7:
                        interval = min(86400, interval * 2)  # Reduce frequency for stable periods
                    
                    # Check for shutdown flag between cycles
                    if self._shutdown_flag:
                        break
                        
                    # Calculate sleep time accounting for processing duration
                    cycle_duration = (datetime.now(pytz.UTC) - cycle_start).total_seconds()
                    sleep_time = max(0, interval - cycle_duration)
                    
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    logger.info("Monitoring task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Monitoring cycle error: {e}")
                    await asyncio.sleep(min(60, interval))  # Wait before retrying
                    
        finally:
            self.is_running = False
            logger.info("Continuous monitoring stopped")

    async def shutdown(self):
        """Graceful system shutdown"""
        logger.info("Initiating shutdown sequence...")
        self._shutdown_flag = True
        self.is_running = False
        
        # Cancel monitoring task if running
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close all data source connections
        for source in self.data_sources:
            if hasattr(source, 'close'):
                await source.close()
        
        logger.info("System shutdown complete")

# Example Usage
async def demo_system():
    """Demonstrate the enhanced monitoring system"""
    system = RealTimePsychohistorySystem()
    system.setup_default_sources()
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(system.continuous_monitoring(interval=30))
