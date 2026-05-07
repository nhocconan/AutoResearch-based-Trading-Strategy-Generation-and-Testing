#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v3"
timeframe = "4h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily Camarilla levels and EMA to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above daily R1 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 2.0
            uptrend = close[i] > ema_34_aligned[i]
            
            if close[i] > r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 with volume and in downtrend
            elif close[i] < s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below daily pivot or volume drops
            if close[i] < pivot_aligned[i] or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above daily pivot or volume drops
            if close[i] > pivot_aligned[i] or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and daily EMA(34) trend filter.
# Uses actual daily Camarilla pivot levels (R1/S1) from prior day's OHLC.
# Breaks above R1 or below S1 with 2x volume indicate institutional interest.
# Daily EMA(34) ensures trades align with daily trend direction.
# Works in bull (buy R1 breaks in uptrend) and bear (sell S1 breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~20-40/year.