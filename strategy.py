#!/usr/bin/env python3
# 1h_4h_1d_Three_Point_Trend_Breakout
# Hypothesis: Uses 1d 3-point trend structure (higher highs/lows or lower highs/lows) to determine trend direction,
# and enters on 1h breakouts of swing points in the direction of the 1d trend. Volume confirmation (>1.5x 20-period average)
# filters for institutional participation. 4h timeframe used for additional trend filtering to reduce noise.
# Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following the 1d structure while using 1h breaks for precise entries.

name = "1h_4h_1d_Three_Point_Trend_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for 3-point trend structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 3-point trend: Higher Highs (HH) and Higher Lows (HL) for uptrend,
    # Lower Highs (LH) and Lower Lows (LL) for downtrend
    # We'll track consecutive HH/HL or LH/LL to establish trend
    hh_hl = np.zeros(len(high_1d), dtype=bool)  # Higher High and Higher Low
    lh_ll = np.zeros(len(high_1d), dtype=bool)  # Lower High and Lower Low
    
    for i in range(2, len(high_1d)):
        # Higher High: current high > previous high
        # Higher Low: current low > previous low
        hh = high_1d[i] > high_1d[i-1]
        hl = low_1d[i] > low_1d[i-1]
        hh_hl[i] = hh and hl
        
        # Lower High: current high < previous high
        # Lower Low: current low < previous low
        lh = high_1d[i] < high_1d[i-1]
        ll = low_1d[i] < low_1d[i-1]
        lh_ll[i] = lh and ll
    
    # Determine trend state: need 2 consecutive HH/HL for uptrend, 2 consecutive LH/LL for downtrend
    uptrend = np.zeros(len(high_1d), dtype=bool)
    downtrend = np.zeros(len(high_1d), dtype=bool)
    
    uptrend_count = 0
    downtrend_count = 0
    
    for i in range(len(high_1d)):
        if hh_hl[i]:
            uptrend_count += 1
            downtrend_count = 0
        elif lh_ll[i]:
            downtrend_count += 1
            uptrend_count = 0
        else:
            uptrend_count = 0
            downtrend_count = 0
        
        uptrend[i] = uptrend_count >= 2
        downtrend[i] = downtrend_count >= 2
    
    # Align 1d trend to 1h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    # 4h data for additional trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(high_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h swing points for entry/exit
    swing_high_1h = np.zeros(len(high), dtype=bool)
    swing_low_1h = np.zeros(len(low), dtype=bool)
    
    for i in range(1, len(high)-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high_1h[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low_1h[i] = True
    
    # Calculate 1h swing high and low levels
    last_swing_high_1h = np.full(len(high), np.nan)
    last_swing_low_1h = np.full(len(low), np.nan)
    
    last_high_1h = np.nan
    last_low_1h = np.nan
    
    for i in range(len(high)):
        if swing_high_1h[i]:
            last_high_1h = high[i]
        if swing_low_1h[i]:
            last_low_1h = low[i]
        last_swing_high_1h[i] = last_high_1h
        last_swing_low_1h[i] = last_low_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(uptrend_1d_aligned[i]) or
            np.isnan(downtrend_1d_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(last_swing_high_1h[i]) or
            np.isnan(last_swing_low_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend on 1d + price above 4h EMA50 + price breaks above 1h swing high + volume spike
            if (uptrend_1d_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                close[i] > last_swing_high_1h[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend on 1d + price below 4h EMA50 + price breaks below 1h swing low + volume spike
            elif (downtrend_1d_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  close[i] < last_swing_low_1h[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 1h swing low OR 1d trend turns down OR price below 4h EMA50
            if (close[i] < last_swing_low_1h[i]) or \
               downtrend_1d_aligned[i] or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 1h swing high OR 1d trend turns up OR price above 4h EMA50
            if (close[i] > last_swing_high_1h[i]) or \
               uptrend_1d_aligned[i] or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals