#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA34 on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly pivot levels (R1, S1 from previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_vals = df_1w['close'].values
    pivot_point = (high_1w + low_1w + close_1w_vals) / 3
    r1_level = 2 * pivot_point - low_1w
    s1_level = 2 * pivot_point - high_1w
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_level)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34 = ema34_1w_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R1 + weekly uptrend + volume spike
            if close[i] > r1 and close[i] > ema34 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S1 + weekly downtrend + volume spike
            elif close[i] < s1 and close[i] < ema34 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S1 or weekly trend turns down
            if close[i] < s1 or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R1 or weekly trend turns up
            if close[i] > r1 or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals