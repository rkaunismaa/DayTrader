
#!/usr/bin/env python3
"""
Real-Time Financial Sentiment Monitor
Runs as a background service, continuously analyzing news.

Usage:
    python sentiment_monitor.py --watchlist NVDA,TSLA,AAPL --interval 30
"""

import time
import json
import threading
import queue
from datetime import datetime
from collections import defaultdict

import feedparser
import torch
from transformers import pipeline


class SentimentMonitor:
    def __init__(self, watchlist, batch_interval=5, alert_threshold=0.5):
        self.watchlist = watchlist
        self.batch_interval = batch_interval  # seconds between processing batches
        self.alert_threshold = alert_threshold
        self.headline_queue = queue.Queue()
        self.sentiment_history = defaultdict(list)  # ticker -> [(timestamp, score)]
        self.seen_headlines = set()  # dedup
        
        # Load models
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_model = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert",
            device=device, truncation=True, max_length=512,
        )
    
    def fetch_loop(self, feeds, interval=30):
        """Continuously fetch headlines from RSS feeds."""
        while True:
            for name, url in feeds.items():
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:20]:
                        title = entry.get("title", "")
                        if title and title not in self.seen_headlines:
                            self.seen_headlines.add(title)
                            self.headline_queue.put({
                                "title": title,
                                "source": name,
                                "timestamp": datetime.now().isoformat(),
                            })
                except Exception as e:
                    print(f"Feed error ({name}): {e}")
            time.sleep(interval)
    
    def process_loop(self):
        """Process queued headlines in batches on GPU."""
        while True:
            batch = []
            # Collect headlines for batch_interval seconds
            deadline = time.time() + self.batch_interval
            while time.time() < deadline:
                try:
                    item = self.headline_queue.get(timeout=0.5)
                    batch.append(item)
                except queue.Empty:
                    continue
            
            if not batch:
                continue
            
            # Batch sentiment analysis on GPU
            texts = [item["title"] for item in batch]
            results = self.sentiment_model(texts, batch_size=32)
            
            for item, result in zip(batch, results):
                score = self._normalize(result)
                tickers = self._extract_tickers(item["title"])
                
                for ticker in tickers:
                    self.sentiment_history[ticker].append(
                        (item["timestamp"], score)
                    )
                    
                    # Alert on strong sentiment
                    if abs(score) > self.alert_threshold:
                        direction = "BULLISH" if score > 0 else "BEARISH"
                        print(f"[ALERT] {direction} {ticker} ({score:+.2f}): {item['title'][:60]}")
    
    def _normalize(self, result):
        label = result["label"].lower()
        score = result["score"]
        if "positive" in label:
            return score
        elif "negative" in label:
            return -score
        return 0.0
    
    def _extract_tickers(self, text):
        # Simplified keyword matching (use the full version from notebook)
        found = []
        for ticker in self.watchlist:
            if ticker in text.upper():
                found.append(ticker)
        return found
    
    def get_current_sentiment(self, ticker, window_minutes=60):
        """Get average sentiment for a ticker over the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [
            score for ts, score in self.sentiment_history[ticker]
            if datetime.fromisoformat(ts) > cutoff
        ]
        if not recent:
            return 0.0, 0
        return sum(recent) / len(recent), len(recent)
    
    def run(self, feeds):
        """Start the monitor with fetch and process threads."""
        fetch_thread = threading.Thread(
            target=self.fetch_loop, args=(feeds, 30), daemon=True
        )
        process_thread = threading.Thread(
            target=self.process_loop, daemon=True
        )
        
        fetch_thread.start()
        process_thread.start()
        
        print("Sentiment monitor running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                # Print periodic summary
                print(f"\n--- Sentiment Summary ({datetime.now().strftime('%H:%M')}) ---")
                for ticker in self.watchlist:
                    avg, count = self.get_current_sentiment(ticker)
                    if count > 0:
                        print(f"  {ticker}: {avg:+.3f} ({count} headlines in last hour)")
        except KeyboardInterrupt:
            print("Monitor stopped.")
