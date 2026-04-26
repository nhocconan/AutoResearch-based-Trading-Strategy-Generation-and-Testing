#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend_v1
Hypothesis: Camarilla R1/S1 breakout with volume spike and 1d EMA34 trend filter for 4h timeframe.
Only long when price breaks above R1 with volume spike and close > 1d EMA34.
Only short when price breaks below S1 with volume spike and close < 1d EMA34.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
Works in both bull and bear markets by combining price structure (Camarilla) with volume confirmation and trend filter.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Need at least 2 days of data for Camarilla calculation
        if i < 2:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        # Camarilla levels use previous day's high, low, close
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla R1, S1 levels
        R1 = ((high[i-1] - low[i-1]) * 1.1 / 12) + close[i-1]
        S1 = close[i-1] - ((high[i-1] - low[i-1]) * 1.1 / 12)
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R1 + volume spike + price > 1d EMA34 (uptrend)
        if close[i] > R1 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume spike + price < 1d EMA34 (downtrend)
        elif close[i] < S1 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: loss of volume spike or trend reversal
        elif position == 1 and (not volume_spike[i] or close[i] < ema_34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not volume_spike[i] or close[i] > ema_34_1d_aligned[i]):
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

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend_v1"
timeframe = "4h"
leverage = 1.0