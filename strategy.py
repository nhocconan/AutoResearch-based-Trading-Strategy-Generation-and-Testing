#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Only long when price breaks above R1 and close > 1d EMA34, short when price breaks below S1 and close < 1d EMA34.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 75-200 trades over 4 years (19-50/year).
Works in both bull and bear markets by combining price action (Camarilla breakouts) with trend (1d EMA) and volume filters.
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
    
    # Calculate Camarilla levels from previous day
    # Typical price of previous day: (high + low + close) / 3
    # We need to get previous day's data - using 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].values[:-1]  # Shift to get previous day
    prev_low = df_1d['low'].values[:-1]
    prev_close = df_1d['close'].values[:-1]
    
    # Calculate typical price and range
    typical_price = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = typical_price + (range_hl * 1.1 / 12)
    S1 = typical_price - (range_hl * 1.1 / 12)
    
    # Align to 4h timeframe (previous day's levels apply to current day)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R1 + close > 1d EMA34 + volume spike
        if close[i] > R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + close < 1d EMA34 + volume spike
        elif close[i] < S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to typical price level or loss of volume confirmation
        elif position == 1 and (close[i] < typical_price[len(typical_price)-len(prices)+i] if len(typical_price) >= len(prices) and len(typical_price)-len(prices)+i >= 0 else close[i] < ema_34_1d_aligned[i]) or not volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > typical_price[len(typical_price)-len(prices)+i] if len(typical_price) >= len(prices) and len(typical_price)-len(prices)+i >= 0 else close[i] > ema_34_1d_aligned[i]) or not volume_spike[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0