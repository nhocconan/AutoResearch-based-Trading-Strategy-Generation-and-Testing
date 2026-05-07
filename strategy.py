#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA50 trend filter and volume confirmation capture high-probability breakout trades in trending markets. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend). Low-frequency signals via 12h timeframe and confluence of Camarilla levels, trend, and volume.
"""
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    pivot = (high + low + close) / 3.0
    r1 = close + (high - low) * 1.1 / 12.0
    s1 = close - (high - low) * 1.1 / 12.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above R1 + 1d uptrend + volume
            if close[i] > r1[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 + 1d downtrend + volume
            elif close[i] < s1[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite Camarilla level
            if position == 1:
                if close[i] < s1[i]:  # Exit long when price breaks below S1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1[i]:  # Exit short when price breaks above R1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals