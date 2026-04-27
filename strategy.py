#!/usr/bin/env python3
"""
1d_ThreeSigmaBreakout_1wTrend_Filter
Hypothesis: On 1d timeframe, enter long when price breaks above 3-sigma upper band (mean + 3*std) and short when breaks below 3-sigma lower band (mean - 3*std) of 20-day price distribution, only when 1-week trend confirms direction (price > 1-week EMA20 for longs, < for shorts). Exit on opposite band touch. Uses volatility-adjusted breakouts to avoid false signals in low-vol regimes and trend filter to avoid counter-trend whipsaws. Designed for low trade frequency (<25/year) to minimize fee drag, works in both bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 3-sigma bands from 1d timeframe (20-day mean and std)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day rolling mean and std dev
    close_1d = df_1d['close'].values
    mean_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    
    # 3-sigma bands: mean ± 3*std
    upper_band = mean_20 + 3.0 * std_20
    lower_band = mean_20 - 3.0 * std_20
    
    # Align bands to 1d timeframe (wait for previous day's close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    mean_20_aligned = align_htf_to_ltf(prices, df_1d, mean_20)  # for exit reference
    
    # 1-week EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for 20-day stats and 1w EMA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(mean_20_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        mean_val = mean_20_aligned[i]
        ema_1w_val = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND above 1w EMA20 (uptrend)
            if close[i] > upper_val and close[i] > ema_1w_val:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band AND below 1w EMA20 (downtrend)
            elif close[i] < lower_val and close[i] < ema_1w_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches or crosses below mean (mean reversion)
            if close[i] < mean_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches or crosses above mean (mean reversion)
            if close[i] > mean_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ThreeSigmaBreakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0