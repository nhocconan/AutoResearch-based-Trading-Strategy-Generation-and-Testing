#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Use 1d CAMARILLA pivot levels (R1/S1) with trend filter from 1d EMA34 and volume confirmation. Enter long when price breaks above R1 in uptrend with volume spike, short when price breaks below S1 in downtrend with volume spike. Exit when price returns to pivot point (PP). Designed for 4h timeframe to capture multi-day moves while minimizing trade frequency. Works in bull/bear via trend filter.
"""

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get daily data for CAMARILLA pivot and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Previous day's OHLC for CAMARILLA calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate CAMARILLA pivot levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = pp + (prev_high - prev_low) * 1.1 / 12  # R1
    s1 = pp - (prev_high - prev_low) * 1.1 / 12  # S1
    
    # Align CAMARILLA levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend determination using EMA34
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals