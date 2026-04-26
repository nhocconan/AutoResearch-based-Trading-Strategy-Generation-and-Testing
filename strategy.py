#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime
Hypothesis: Camarilla R1/S1 breakout on 12h with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above R1 with volume > 1.5x median volume and close > 1d EMA50.
Short when price breaks below S1 with volume confirmation and close < 1d EMA50.
Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
Works in both bull and bear markets by following the 1d trend while using 12h breakouts for precise entries.
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
    
    # Calculate Camarilla levels for 12h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume median, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume confirmation and 1d uptrend
        long_condition = (close[i] > r1[i]) and volume_confirm[i] and (close[i] > ema_50_1d_aligned[i])
        # Short logic: break below S1 with volume confirmation and 1d downtrend
        short_condition = (close[i] < s1[i]) and volume_confirm[i] and (close[i] < ema_50_1d_aligned[i])
        
        # Exit logic: opposite Camarilla level touch or trend reversal
        exit_long = (close[i] < s1[i]) or (close[i] < ema_50_1d_aligned[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_50_1d_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0