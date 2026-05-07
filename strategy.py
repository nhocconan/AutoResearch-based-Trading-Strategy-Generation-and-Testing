#!/usr/bin/env python3
name = "4h_1d_1w_Camarilla_R1S1_Breakout_TrendVolume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Camarilla
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    high_low_1d = prev_high_1d - prev_low_1d
    
    # 1d Camarilla levels
    r1_1d = prev_close_1d + (high_low_1d * 1.1 / 12)
    s1_1d = prev_close_1d - (high_low_1d * 1.1 / 12)
    
    # Calculate 1w Camarilla levels from previous week
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    high_low_1w = prev_high_1w - prev_low_1w
    
    # 1w Camarilla levels
    r1_1w = prev_close_1w + (high_low_1w * 1.1 / 12)
    s1_1w = prev_close_1w - (high_low_1w * 1.1 / 12)
    
    # Align levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 3-period average (3*4h = 12h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 3)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 (1d OR 1w) with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if ((close[i] > s1_1d_aligned[i] or close[i] > s1_1w_aligned[i]) and 
                vol_condition and uptrend):
                signals[i] = 0.30
                position = 1
            # Short: price below R1 (1d OR 1w) with volume and daily downtrend
            elif ((close[i] < r1_1d_aligned[i] or close[i] < r1_1w_aligned[i]) and 
                  vol_condition and not uptrend):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S1 (both 1d and 1w) or volume drops
            if close[i] < s1_1d_aligned[i] and close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 (both 1d and 1w) or volume drops
            if close[i] > r1_1d_aligned[i] and close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout from 1d OR 1w with 1d trend and volume confirmation
# - Uses Camarilla levels from BOTH daily and weekly timeframes for confluence
# - Breakout above S1 (either timeframe) with volume in daily uptrend = long
# - Breakdown below R1 (either timeframe) with volume in daily downtrend = short
# - Volume spike (2.0x 3-period average) confirms institutional participation
# - Requires BOTH 1d and 1w levels to be broken for exit (more stringent)
# - Works in bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Position size 0.30 targets ~25-40 trades/year, avoiding fee drag
# - More robust than single timeframe due to dual timeframe confirmation
# - Volume confirmation reduces false breakouts in choppy markets
# - Dual timeframe approach not recently tried in this combination