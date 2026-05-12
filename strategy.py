#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volume confirmation.
# Works in bull/bear by following 1d trend direction. R1/S1 are key intraday support/resistance levels.
# Volume ensures breakouts have conviction. Designed for low trade frequency (<400 total) to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # === 1d EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Camarilla Levels (based on previous day) ===
    # For 4h bars, we use daily Camarilla levels calculated from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_level = close_1d + camarilla_range
    s1_level = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r1_level_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_level_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # === Volume Confirmation (20-period average on 4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1, uptrend, volume confirmation
            if close[i] > r1_level_aligned[i] and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, downtrend, volume confirmation
            elif close[i] < s1_level_aligned[i] and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend changes
            if close[i] < r1_level_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend changes
            if close[i] > s1_level_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals