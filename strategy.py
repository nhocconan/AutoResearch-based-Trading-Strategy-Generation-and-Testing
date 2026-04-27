#!/usr/bin/env python3
"""
1d_KAMA_Trend_WeeklyVolumeSpike_ExitOnReverse
Hypothesis: Uses KAMA (10,2,30) on daily timeframe for trend direction, with weekly volume spike confirmation for entry, and exits on trend reversal. KAMA adapts to market noise, reducing whipsaws in sideways markets. Weekly volume spike (>2.0x 20-week average) confirms institutional interest. Designed to work in both bull (trend following) and bear (avoiding false breakouts via volume filter) markets. Target: 20-80 total trades over 4 years (5-20/year) with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA on daily close: ER = 10, Fast = 2, Slow = 30
    close_1d_series = pd.Series(df_1d['close'].values)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close_1d_series - close_1d_series.shift(10))
    volatility = close_1d_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)  # avoid div by zero
    # Smoothing constants: sc = [ER*(2/(2+1)-2/(30+1)) + 2/(30+1)]^2
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA: kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
    kama = np.full_like(close_1d_series, np.nan, dtype=float)
    kama[9] = close_1d_series.iloc[9]  # seed after 10 periods
    for i in range(10, len(close_1d_series)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d_series.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_values = kama.values
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # Weekly volume spike: current weekly volume > 2.0 * 20-week average
    vol_1w_series = pd.Series(df_1w['volume'].values)
    vol_avg_1w = vol_1w_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_avg_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10 + 1 seed), volume avg (20)
    start_idx = max(30, 20)  # KAMA needs ~30 to stabilize, volume 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Look for entry: price crosses KAMA with weekly volume spike
            # Long: price crosses above KAMA AND volume spike
            long_condition = (close_val > kama_val) and vol_spike
            # Short: price crosses below KAMA AND volume spike
            short_condition = (close_val < kama_val) and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price crosses below KAMA (trend reversal)
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price crosses above KAMA (trend reversal)
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_WeeklyVolumeSpike_ExitOnReverse"
timeframe = "1d"
leverage = 1.0