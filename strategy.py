# 2025-06-22 18:02:00
#!/usr/bin/env python3
"""
4h_Camarilla_R2_S2_Breakout_1dTrend_Volume_Spread
Hypothesis: Trading breakouts of wider Camarilla R2/S2 levels with daily trend filter and volume confirmation.
Wider levels reduce false breakouts, lowering trade frequency while maintaining edge.
Works in bull via uptrend longs, in bear via downtrend shorts. Target: 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla R2 and S2 (wider levels for fewer, stronger breakouts)
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align to 4h timeframe (use previous day's levels)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r2_val = r2_1d_aligned[i]
        s2_val = s2_1d_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R2 with uptrend and volume spike
            if close[i] > r2_val and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S2 with downtrend and volume spike
            elif close[i] < s2_val and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below S2 or trend turns down
            if close[i] < s2_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R2 or trend turns up
            if close[i] > r2_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_1dTrend_Volume_Spread"
timeframe = "4h"
leverage = 1.0