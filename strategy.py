#!/usr/bin/env python3
# 4h_Three_Stage_Trend_Filter
# Hypothesis: Combines 1-day trend filter, 1-week regime filter, and 4-hour price action with volume confirmation.
# Long when: price > 1d EMA34, price > 1w EMA89 (bull regime), and price > 4h KAMA with volume spike.
# Short when: price < 1d EMA34, price < 1w EMA89 (bear regime), and price < 4h KAMA with volume spike.
# Uses weekly EMA to distinguish bull/bear regimes, reducing whipsaw in sideways markets.
# Target: 20-40 trades/year with strong trend signals only.

name = "4h_Three_Stage_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for multi-timeframe filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 34 or len(df_1w) < 89:
        return np.zeros(n)
    
    # Calculate 4h KAMA (adaptive moving average)
    # Efficiency Ratio: |price change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(abs_change).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants for KAMA
    fast_sc = 2 / (2 + 1)   # EMA 2
    slow_sc = 2 / (30 + 1)  # EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA89 for regime filter (bull/bear detection)
    ema_89_1w = pd.Series(df_1w['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need sufficient data for all indicators
    start_idx = max(10, 34, 89, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_89_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Multi-timeframe trend alignment
        # Bull regime: price above both daily and weekly EMA
        bull_regime = (close[i] > ema_34_1d_aligned[i]) and (close[i] > ema_89_1w_aligned[i])
        # Bear regime: price below both daily and weekly EMA
        bear_regime = (close[i] < ema_34_1d_aligned[i]) and (close[i] < ema_89_1w_aligned[i])
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to 4h KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long entry: bull regime + price above KAMA + volume spike
            if bull_regime and price_above_kama and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bear regime + price below KAMA + volume spike
            elif bear_regime and price_below_kama and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: regime turns bearish or price crosses below KAMA
            if bear_regime or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: regime turns bullish or price crosses above KAMA
            if bull_regime or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals