#!/usr/bin/env python3
"""
4h KAMA Trend with Volume Confirmation and ADX Filter
Hypothesis: KAMA adapts to market conditions, reducing false signals in choppy markets.
Combined with volume confirmation and ADX trend filter, it captures strong trends in both bull and bear markets while avoiding whipsaws. Target: 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average)
    close_series = pd.Series(close)
    direction = np.abs(close_series.diff(10).values)
    volatility = np.abs(close_series.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ADX for trend strength (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA, ADX, and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        adx_val = adx[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA with rising ADX and volume
            if price > kama_val and adx_val > 25 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with rising ADX and volume
            elif price < kama_val and adx_val > 25 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below KAMA or ADX weakens
            if price < kama_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above KAMA or ADX weakens
            if price > kama_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0