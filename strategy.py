#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Pullback_1dTrend_VolumeConfirm
Hypothesis: In strong daily trends (price > EMA50 for long, price < EMA50 for short), 
wait for pullback to Camarilla S1 (long) or R1 (short) with volume confirmation.
This trades in the direction of the daily trend but enters on retracements, 
reducing false breakouts and improving win rate. Works in bull via trend continuation 
pullbacks and in bear by avoiding counter-trend entries via daily EMA filter.
Discrete sizing (0.25) and tight entry conditions target 75-200 trades over 4 years.
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
    
    # Calculate Camarilla levels for 4h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 1.8x 20-period median (robust to outliers)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.8)
    
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
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: pullback to S1 in daily uptrend with volume spike
        long_condition = (close[i] <= s1[i] * 1.002) and (close[i] >= s1[i] * 0.998) and \
                         volume_spike[i] and (close[i] > ema_50_1d_aligned[i])
        # Short logic: pullback to R1 in daily downtrend with volume spike
        short_condition = (close[i] >= r1[i] * 0.998) and (close[i] <= r1[i] * 1.002) and \
                          volume_spike[i] and (close[i] < ema_50_1d_aligned[i])
        
        # Exit logic: price crosses EMA50 (trend change) or reaches opposite Camarilla level
        exit_long = (close[i] < ema_50_1d_aligned[i]) or (close[i] >= r1[i])
        exit_short = (close[i] > ema_50_1d_aligned[i]) or (close[i] <= s1[i])
        
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

name = "4h_Camarilla_R1_S1_Pullback_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0