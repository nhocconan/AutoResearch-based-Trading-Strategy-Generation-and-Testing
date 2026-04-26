#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA34 trend filter and volume spike confirmation (>2.0x median).
Designed for lower trade frequency (target: 12-37/year) to minimize fee drift. Works in bull via breakout continuation,
in bear by avoiding counter-trend trades via daily EMA filter. Uses discrete sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
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
    
    # Volume confirmation: volume > 2.0x 20-period median (robust to outliers)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 20-period for volume median, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume spike and 1d uptrend
        long_condition = (close[i] > r1[i]) and volume_spike[i] and (close[i] > ema_34_1d_aligned[i])
        # Short logic: break below S1 with volume spike and 1d downtrend
        short_condition = (close[i] < s1[i]) and volume_spike[i] and (close[i] < ema_34_1d_aligned[i])
        
        # Exit logic: opposite Camarilla level (S1/R1) or trend reversal
        exit_long = (close[i] < s1[i]) or (close[i] < ema_34_1d_aligned[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_34_1d_aligned[i])
        
        # Minimum holding period: 4 bars (~2 days) to reduce churn
        if position != 0 and bars_since_entry < 4:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0