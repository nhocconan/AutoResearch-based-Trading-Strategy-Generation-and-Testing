#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1D_Volume_Spike"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1D data ONCE for Camarilla pivot, volume, and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla pivot levels (R1, S1)
    # Using previous day's data to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Calculate volume spike (current volume > 2x 20-period average)
    vol_series = pd.Series(volume_1d)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2 * vol_ma)
    
    # Calculate 1D EMA50 for trend filter
    close_series = pd.Series(close_1d)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to 12H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and uptrend
            if close[i] > r1_aligned[i] and volume_spike_aligned[i] and close[i] > ema50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and downtrend
            elif close[i] < s1_aligned[i] and volume_spike_aligned[i] and close[i] < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or volume dries up
            if close[i] < s1_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or volume dries up
            if close[i] > r1_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals