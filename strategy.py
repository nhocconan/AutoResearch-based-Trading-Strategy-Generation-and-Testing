#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels from 1d for precise entry/exit, with 1d trend filter and volume confirmation.
# Enter long when price breaks above R1 level with volume in uptrend, short when breaks below S1 level with volume in downtrend.
# Exit when price reaches opposite S1/R1 level or trend reverses.
# Designed for low frequency (12-30 trades/year) by using 12h for signal and 1d for trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1d data for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # R2 = close + ((high - low) * 1.166)
    # R1 = close + ((high - low) * 1.083)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.083)
    # S2 = close - ((high - low) * 1.166)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    hl_range = high_1d - low_1d
    r1_1d = close_1d + (hl_range * 1.083)
    s1_1d = close_1d - (hl_range * 1.083)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12h (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # === 1d data for trend filter ===
    # EMA(34) on 1d for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, in uptrend
            if close[i] > r1_1d_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume, in downtrend
            elif close[i] < s1_1d_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches S1 level or trend reverses
            if close[i] <= s1_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches R1 level or trend reverses
            if close[i] >= r1_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals