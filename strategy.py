#!/usr/bin/env python3
# 4h_Cam_R1S1_Breakout_1dTrend_Volume_Spike
# Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Works in bull markets via R1 breakouts and in bear markets via S1 breakdowns
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades
# Target: 25-50 trades per year (~100-200 over 4 years) with position size 0.25

name = "4h_Cam_R1S1_Breakout_1dTrend_Volume_Spike"
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
    
    # Load 1d data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (for current 4h bar)
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    r1 = close_prev + 1.1 * (high_prev - low_prev) / 2
    s1 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 bar for Camarilla calculation
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]   # Break above R1
        breakout_down = close[i] < s1_aligned[i]  # Break below S1
        
        # Volume confirmation: volume > 2.0x average (strict filter)
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: R1 breakout + volume spike + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: S1 breakdown + volume spike + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 or trend reversal
            if close[i] < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 or trend reversal
            if close[i] > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals