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

# Import our enhanced psychohistory framework
from enhanced_psychohistory_framework import PsychohistoryEngine, CivilizationMetrics, MetricCategory

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
            start_time = datetime.now(pytz.UTC)
            data = await self.fetch_data()
            
            # Process and validate data
            processed = self.process_raw_data(data)
            if not processed:
                raise ValueError("No data points returned")
                
            # Update health metrics
            self._consecutive_errors = 0
            self.health_score = min(1.0, self.health_score + 0.1)
            
            # Adjust interval based on data volatility
            if len(processed) > 0:
                values = [dp.value for dp in processed]
                volatility = np.std(values)
                # More volatile data -> more frequent updates
                self._adaptive_interval = max(300, min(
                    self.update_frequency * 3,
                    int(self.update_frequency / (1 + volatility * 2)
                ))
            
            return processed
            
        except Exception as e:
            self._consecutive_errors += 1
            self.health_score = max(0.1, self.health_score - 0.2)
            logger.error(f"Error in {self.name} source: {str(e)}")
            
            # Exponential backoff on repeated errors
            if self._consecutive_errors > 2:
                self._adaptive_interval = min(
                    86400,  # Max 1 day
                    self.update_frequency * (2 ** self._consecutive_errors)
                )
            
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
        self.session = aiohttp.ClientSession()
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
    
    async def fetch_data(self) -> Any:
        """Default API fetch implementation with rate limiting"""
        if self._rate_limit_remaining <= 1:
            reset_time = datetime.now(pytz.UTC) + timedelta(seconds=self._rate_limit_reset)
            if datetime.now(pytz.UTC) < reset_time:
                wait_time = (reset_time - datetime.now(pytz.UTC)).total_seconds()
                logger.warning(f"Rate limited - waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with self.session.get(self.base_url, headers=headers) as response:
            # Update rate limit tracking
            self._rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
            self._rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 60))
            
            if response.status != 200:
                raise ValueError(f"API request failed: {response.status}")
            
            return await response.json()
    
    async def close(self):
        """Clean up resources"""
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
        """Fetch sentiment data from API"""
        # In a real implementation, this would make authenticated API calls
        # Simulating API response with realistic data patterns
        
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
                "ai_fear_sentiment": max(0, min(1, (1 - base_sentiment) * np.random.beta(4, 4))),
                "information_trust_level": max(0, min(1, base_sentiment * np.random.beta(2, 6))),
            },
            "metadata": {
                "sample_size": np.random.randint(10000, 50000),
                "platforms": ["twitter", "reddit", "facebook"]
            }
        }
    
    def process_raw_data(self, raw_data: Any) -> List[DataPoint]:
        """Process API response with anomaly detection"""
        data_points = []
        current_time = datetime.now(pytz.UTC)
        
        # Track anomalies across metrics
        anomalies_detected = 0
        
        for metric, value in raw_data["metrics"].items():
            # Calculate confidence based on sample size
            sample_size = raw_data["metadata"]["sample_size"]
            confidence = min(0.99, 0.7 + (sample_size / 50000) * 0.3)
            
            # Simple anomaly detection (in real system would use historical data)
            expected_range = {
                "institutional_trust_sentiment": (0.1, 0.4),
                "economic_anxiety_level": (0.4, 0.8),
                "political_polarization_index": (0.6, 0.9),
                "social_cohesion_sentiment": (0.3, 0.6),
                "future_optimism_index": (0.2, 0.5),
                "ai_fear_sentiment": (0.3, 0.7),
                "information_trust_level": (0.1, 0.4)
            }
            
            lower, upper = expected_range.get(metric, (0.0, 1.0))
            is_anomaly = not (lower <= value <= upper)
            
            if is_anomaly:
                anomalies_detected += 1
                quality = DataQuality.SUSPECT
                confidence *= 0.7  # Reduce confidence for anomalies
            else:
                quality = DataQuality.TRUSTED
            
            data_points.append(DataPoint(
                source=self.name,
                metric=metric,
                value=value,
                timestamp=current_time,
                confidence=confidence,
                quality=quality,
                metadata={
                    **raw_data["metadata"],
                    "is_anomaly": is_anomaly
                }
            ))
        
        # If multiple anomalies detected, log warning
        if anomalies_detected >= 3:
            logger.warning(f"Multiple anomalies detected in social sentiment data: {anomalies_detected}")
        
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
        
        # Initialize database with better performance settings
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
        
        # Additional sources would be added here
        # self.add_data_source(EconomicIndicatorSource())
        # self.add_data_source(AIAdoptionTracker())
        # etc...
    
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
        
        # Gather tasks for all sources due for update
        tasks = []
        for source in self.data_sources:
            if source.is_active and source.next_update_due:
                tasks.append(source.adaptive_fetch())
        
        # Process results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                data_points = await task
                if data_points:
                    await self._store_data_points(data_points)
                    total_points += len(data_points)
            except Exception as e:
                logger.error(f"Error in data collection task: {e}")
        
        return total_points
    
    async def stream_data_points(self, metric_filter: str = None) -> AsyncGenerator[DataPoint, None]:
        """Stream new data points as they arrive (for real-time dashboards)"""
        last_id = None
        
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
                    
                    # Mark as processed
                    cursor.execute(
                        "UPDATE data_points SET processed = 1 WHERE id = ?",
                        (row[0],)
                    
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
        """Get weighted average metrics from recent data"""
        metrics = CivilizationMetrics()
        window_hours = 24  # Look at last 24 hours of data
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get most recent data points for each metric with weighting
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
            
            # Apply to metrics object
            metric_mapping = self._get_metric_mapping()
            
            for metric, readings in metric_groups.items():
                if metric not in metric_mapping:
                    continue
                    
                category, field = metric_mapping[metric]
                category_dict = getattr(metrics, category)
                
                # Calculate weighted average (confidence * quality)
                total_weight = 0
                weighted_sum = 0
                
                for reading in readings[:10]:  # Limit to 10 most recent readings
                    weight = reading['confidence'] * (reading['quality'] / 3.0)
                    weighted_sum += reading['value'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    category_dict[field] = weighted_sum / total_weight
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return metrics
        finally:
            conn.close()
    
    def _get_metric_mapping(self) -> Dict[str, tuple]:
        """Get mapping from source metrics to our framework metrics"""
        return {
            # Economic indicators
            'wealth_inequality_gini': ('economic_indicators', 'wealth_inequality'),
            'currency_volatility_index': ('economic_indicators', 'currency_stability'),
            'debt_to_gdp_ratio': ('economic_indicators', 'debt_to_gdp'),
            'inflation_rate': ('economic_indicators', 'inflation_rate'),
            
            # Social indicators  
            'social_cohesion_sentiment': ('social_indicators', 'civic_engagement'),
            'future_optimism_index': ('social_indicators', 'social_mobility'),
            
            # Political indicators
            'institutional_trust': ('political_indicators', 'institutional_trust'),
            'institutional_trust_sentiment': ('political_indicators', 'institutional_trust'),
            'corruption_perception': ('political_indicators', 'corruption_index'),
            'political_stability_index': ('political_indicators', 'political_stability'),
            'democratic_backsliding': ('political_indicators', 'democratic_index'),
            
            # Environmental indicators
            'resource_depletion_rate': ('environmental_indicators', 'resource_depletion'),
            'climate_stress_index': ('environmental_indicators', 'climate_stress'),
            'agricultural_productivity': ('environmental_indicators', 'agricultural_productivity'),
            
            # AI influence indicators
            'ai_penetration_rate': ('ai_influence_indicators', 'ai_penetration_rate'),
            'cognitive_outsourcing': ('ai_influence_indicators', 'cognitive_outsourcing'),
            'algorithmic_governance': ('ai_influence_indicators', 'algorithmic_governance'),
            'reality_authenticity_crisis': ('ai_influence_indicators', 'reality_authenticity_crisis'),
            'human_ai_symbiosis': ('ai_influence_indicators', 'human_ai_symbiosis'),
            'ai_behavioral_conditioning': ('ai_influence_indicators', 'ai_behavioral_conditioning'),
            'information_velocity': ('ai_influence_indicators', 'information_velocity'),
            'personalized_reality_bubbles': ('ai_influence_indicators', 'personalized_reality_bubbles'),
            'decision_dependency': ('ai_influence_indicators', 'decision_dependency'),
            'collective_intelligence_erosion': ('ai_influence_indicators', 'collective_intelligence_erosion'),
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
                        f"Risk: {risk}, Patterns: {len(results['analysis'].get('pattern_matches', []))}"
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
    monitor_task = asyncio.create_task(system.continuous_monitoring(interval=600))
    
    try:
        # Let it run for a while
        await asyncio.sleep(30)
        
        # Get current status
        analysis = await system.analyze_civilization()
        print(f"\nCurrent Stability: {analysis['analysis']['stability_score']:.2f}")
        print(f"Risk Level: {analysis['analysis']['risk_level']}")
        
        # Stream some data points
        print("\nSample Social Media Data:")
        async for point in system.stream_data_points(metric_filter="political_polarization_index"):
            print(f"{point.timestamp}: {point.metric} = {point.value:.2f} (conf: {point.confidence:.2f})")
            if point.value > 0.8:
                print("  Warning: High polarization detected!")
            break  # Just show one for demo
            
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_system())
