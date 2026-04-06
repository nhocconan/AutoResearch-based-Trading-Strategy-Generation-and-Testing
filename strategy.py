#!/usr/bin/env python3
"""
6h Heikin-Ashi Trend Continuation with 12h KAMA Filter and Volume Spike
Hypothesis: Heikin-Ashi smooths noise to identify true trends, while 12h KAMA adapts to market regime (trending vs ranging).
Volume spikes confirm momentum. In trending markets (KAMA slope aligned), we trade HA continuation.
In ranging markets, we avoid trades. This reduces whipsaws in sideways markets while capturing trends.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ha_kama_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Heikin-Ashi calculation
    ha_close = (high + low + close + open) / 4 if 'open' in prices.columns else (high + low + close + close) / 4
    if 'open' not in prices.columns:
        # Calculate open from close if not available (approximation)
        ha_open = np.zeros(n)
        ha_open[0] = close[0]
        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    else:
        open_price = prices['open'].values
        ha_open = (open_price + close) / 2  # Simplified HA open
    
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
    
    # HA trend: green candle (bullish) if ha_close > ha_open, red if ha_close < ha_open
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # 12h KAMA (adaptive moving average) for regime detection
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio and KAMA calculation
    def kama(price, period=10, fast=2, slow=30):
        if len(price) < period:
            return np.full_like(price, np.nan)
        # Change and volatility
        change = np.abs(np.diff(price, period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1) if len(price) > 1 else np.array([0])
        # Pad volatility to match change length
        volatility = np.concatenate([np.full(period-1, np.nan), volatility[:-period+1]]) if len(volatility) >= period-1 else np.full(len(price), np.nan)
        # Avoid division by zero
        er = np.where(volatility > 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_val = np.full_like(price, np.nan)
        kama_val[period-1] = np.mean(price[:period])
        for i in range(period, len(price)):
            if not np.isnan(sc[i]) and not np.isnan(kama_val[i-1]):
                kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
            else:
                kama_val[i] = kama_val[i-1]
        return kama_val
    
    kama_12h = kama(close_12h, period=10, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # KAMA slope for trend direction (1 if rising, -1 if falling)
    kama_slope = np.diff(kama_12h_aligned, prepend=kama_12h_aligned[0])
    kama_trend = np.where(kama_slope > 0, 1, -1)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(kama_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: HA turns bearish OR KAMA trend turns bearish
            if ha_bearish[i] or kama_trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: HA turns bullish OR KAMA trend turns bullish
            if ha_bullish[i] or kama_trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: HA continuation in direction of KAMA trend with volume
            # Long: HA bullish and KAMA trending up
            if ha_bullish[i] and kama_trend[i] == 1 and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: HA bearish and KAMA trending down
            elif ha_bearish[i] and kama_trend[i] == -1 and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals