#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Adaptive_1dATR_Channel_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d ATR(20) for volatility measurement ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d)
    
    # === 1d ATR(50) for regime detection ===
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # === 12h rolling high/low for channel breakout ===
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    high_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_channel_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), high_channel)
    low_channel_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), low_channel)
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Regime: ATR ratio (20/50) for trending/ranging ===
    atr_ratio = atr20_1d_aligned / (atr50_1d_aligned + 1e-10)
    # Trending: ATR ratio > 1.1, Ranging: ATR ratio < 0.9
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_channel_aligned[i]) or np.isnan(low_channel_aligned[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            is_trending = atr_ratio[i] > 1.1
            is_ranging = atr_ratio[i] < 0.9
            
            if is_trending:
                # Trending: breakout in direction of 12h trend (price > mid-channel)
                mid_channel = (high_channel_aligned[i] + low_channel_aligned[i]) / 2
                long_cond = (close[i] > high_channel_aligned[i] and 
                            close[i] > mid_channel and
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] < low_channel_aligned[i] and 
                             close[i] < mid_channel and
                             volume[i] > vol_ma20[i])
            elif is_ranging:
                # Ranging: mean reversion at channel extremes
                long_cond = (close[i] < low_channel_aligned[i] * 1.02 and  # Near low
                            volume[i] > vol_ma20[i])
                short_cond = (close[i] > high_channel_aligned[i] * 0.98 and  # Near high
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
            # Long exit: reverse signal or volatility expansion
            if (close[i] < low_channel_aligned[i] or  # Breakdown
                atr_ratio[i] > 1.5):  # Volatility spike
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or volatility expansion
            if (close[i] > high_channel_aligned[i] or  # Breakout
                atr_ratio[i] > 1.5):  # Volatility spike
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Adaptive 12h strategy that uses 1d ATR ratio to detect regime (trending/ranging).
# In trending markets: breakout trades in direction of 12h channel (volume-confirmed).
# In ranging markets: mean reversion at channel extremes (support/resistance).
# Uses ATR expansion (>1.5x) as volatility-based exit to avoid whipsaws.
# Designed for low trade frequency (target: 50-150/4 years) to minimize fee drag.
# Works in bull/bear via regime adaptation. Uses discrete sizing (0.25) to reduce churn.