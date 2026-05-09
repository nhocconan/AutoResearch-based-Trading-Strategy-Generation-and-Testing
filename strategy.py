#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily pivot points using daily high/low/close
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    pp_1d = (high_1d + low_1d + close_1d_arr) / 3
    r1_1d = (2 * pp_1d) - low_1d
    s1_1d = (2 * pp_1d) - high_1d
    
    # Use previous period's values to avoid look-ahead
    pp_1d_prev = np.roll(pp_1d, 1)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    pp_1d_prev[0] = np.nan
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_1d_prev[i]) or 
            np.isnan(r1_1d_prev[i]) or np.isnan(s1_1d_prev[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike and above 1d EMA trend
            if close[i] > r1_1d_prev[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S1 with volume spike and below 1d EMA trend
            elif close[i] < s1_1d_prev[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S1 (mean reversion)
            if close[i] < s1_1d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price crosses back above R1
            if close[i] > r1_1d_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals