#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_adaptive_breakout_v1
# Combines adaptive moving average (KAMA) trend with Donchian breakout and volume confirmation.
# Uses daily trend filter to avoid counter-trend trades, reducing whipsaw in choppy markets.
# Designed for low trade frequency (<40/year) with high win rate in both bull and bear markets.
name = "4h_1d_adaptive_breakout_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA trend on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.sum(volatility[np.arange(1, len(close_1d))[:, None] == np.arange(len(volatility))[None, :]], axis=1) + 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    # Align KAMA to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if values not ready
        if np.isnan(kama_4h[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long: price above Donchian high AND above daily KAMA (uptrend)
        if close[i] > high_max[i] and close[i] > kama_4h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price below Donchian low AND below daily KAMA (downtrend)
        elif close[i] < low_min[i] and close[i] < kama_4h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian break
        elif close[i] < low_min[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > high_max[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals