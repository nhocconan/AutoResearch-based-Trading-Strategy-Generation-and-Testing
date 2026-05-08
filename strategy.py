#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_With_Volume_and_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d KAMA for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate KAMA: Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === 1d Chopiness Index for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr14 = np.maximum(high_1d - low_1d,
                      np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                 np.abs(low_1d - np.roll(close_1d, 1))))
    tr14[0] = high_1d[0] - low_1d[0]
    atr14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        atr14[i] = np.mean(tr14[i-13:i+1])
    # Chop calculation
    sum_tr14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        sum_tr14[i] = np.sum(tr14[i-13:i+1])
    max_min = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        max_min[i] = np.max(high_1d[i-13:i+1]) - np.min(low_1d[i-13:i+1])
    chop = np.where(max_min != 0, 100 * np.log10(sum_tr14 / max_min) / np.log10(14), 50)
    chop_1d = chop
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Volume filter ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Chop regime: < 38.2 = trending, > 61.8 = ranging
            is_trending = chop_1d_aligned[i] < 38.2
            is_ranging = chop_1d_aligned[i] > 61.8
            
            if is_trending:
                # Trending: KAMA breakout
                long_cond = (close[i] > kama_1d_aligned[i] and
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] < kama_1d_aligned[i] and
                             volume[i] > vol_ma20[i])
            elif is_ranging:
                # Ranging: mean reversion at KAMA
                long_cond = (close[i] < kama_1d_aligned[i] and
                            close[i] > np.min(low[max(0, i-5):i+1]) and  # near recent low
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] > kama_1d_aligned[i] and
                             close[i] < np.max(high[max(0, i-5):i+1]) and  # near recent high
                             volume[i] > vol_ma20[i])
            else:
                # Transition zone: no trades
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit
            if chop_1d_aligned[i] < 38.2:
                # Trending: exit on KAMA breakdown
                exit_cond = close[i] < kama_1d_aligned[i]
            else:
                # Ranging: exit on reversion to KAMA
                exit_cond = close[i] > kama_1d_aligned[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit
            if chop_1d_aligned[i] < 38.2:
                # Trending: exit on KAMA breakout
                exit_cond = close[i] > kama_1d_aligned[i]
            else:
                # Ranging: exit on reversion to KAMA
                exit_cond = close[i] < kama_1d_aligned[i]
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA-based trend strategy with Chop regime filter.
# Uses daily KAMA as trend filter and Chop index to switch between
# trend-following (breakout) in trending markets and mean-reversion
# in ranging markets. Volume confirmation ensures participation.
# Designed to work in both bull (trend following) and bear (mean reversion in ranges)
# markets. Targets 80-150 trades over 4 years (20-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via institutional
# trend and regime detection. Avoids overtrading by using clear regime thresholds.