#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume Spike + 1d EMA50 Trend + Camarilla R4/S4 Breakout
# Long when: price breaks above R4 AND close > 1d EMA50 AND volume > 2.0x 20-period average
# Short when: price breaks below S4 AND close < 1d EMA50 AND volume > 2.0x 20-period average
# Exit: price reverts to Camarilla pivot point (PP) OR EMA50 filter reverses
# Uses wider Camarilla bands (R4/S4) for fewer, higher-quality breakouts
# 1d EMA50 provides strong trend filter effective in both bull and bear markets
# Volume spike (2.0x) reduces false signals and controls trade frequency
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1d

name = "12h_Camarilla_R4S4_Breakout_1dEMA50_VolumeSpike_2.0x"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_high = np.roll(high_1d, 1)  # previous day high
    camarilla_low = np.roll(low_1d, 1)    # previous day low
    camarilla_close = np.roll(close_1d, 1) # previous day close
    
    # Calculate Camarilla levels
    camarilla_range = camarilla_high - camarilla_low
    r4 = camarilla_close + 1.5 * camarilla_range
    s4 = camarilla_close - 1.5 * camarilla_range
    pp = (camarilla_high + camarilla_low + camarilla_close) / 3.0
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 AND price > EMA50 AND volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND price < EMA50 AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to PP OR price < EMA50 (trend weakening)
            if close[i] < pp_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to PP OR price > EMA50 (trend weakening)
            if close[i] > pp_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals