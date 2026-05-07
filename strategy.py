#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with volume confirmation and trend filter.
# Long when: Price breaks above Camarilla R4 + Volume > 1.5x 20-period average + 1d EMA(50) rising
# Short when: Price breaks below Camarilla S4 + Volume > 1.5x 20-period average + 1d EMA(50) falling
# Exit when price crosses back to Camarilla Pivot point.
# Uses 1d for pivot levels and trend filter to avoid lower timeframe noise.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag.
# Works in bull markets via R4 breakouts in uptrend, in bear markets via S4 breakdowns in downtrend.
name = "4h_Camarilla_R4S4_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    R4 = close_1d + range_hl * 1.1 / 2
    R3 = close_1d + range_hl * 1.1 / 4
    R2 = close_1d + range_hl * 1.1 / 6
    R1 = close_1d + range_hl * 1.1 / 12
    S1 = close_1d - range_hl * 1.1 / 12
    S2 = close_1d - range_hl * 1.1 / 6
    S3 = close_1d - range_hl * 1.1 / 4
    S4 = close_1d - range_hl * 1.1 / 2
    P = pivot  # Pivot point
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_rising[1:] = ema_50_1d[1:] > ema_50_1d[:-1]
    ema_50_falling[1:] = ema_50_1d[1:] < ema_50_1d[:-1]
    
    # Align all 1d indicators to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(P_aligned[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 + Volume confirmation + 1d EMA50 rising
            long_condition = (close[i] > R4_aligned[i]) and (vol_ratio[i] > 1.5) and ema_50_rising_aligned[i]
            # Short: Price breaks below S4 + Volume confirmation + 1d EMA50 falling
            short_condition = (close[i] < S4_aligned[i]) and (vol_ratio[i] > 1.5) and ema_50_falling_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses back to Pivot point
            if close[i] < P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses back to Pivot point
            if close[i] > P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals