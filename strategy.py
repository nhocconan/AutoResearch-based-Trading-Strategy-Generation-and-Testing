#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Price breaking Camarilla R1 (resistance 1) or S1 (support 1) levels with volume confirmation
and 1d EMA34 trend filter captures institutional breakouts in both bull and bear markets.
The 1d EMA34 provides robust trend filtering while Camarilla levels offer precise entry/exit points.
Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
"""

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
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4),
    # R2 = close + ((high-low)*1.1/6), R1 = close + ((high-low)*1.1/12)
    # S1 = close - ((high-low)*1.1/12), S2 = close - ((high-low)*1.1/6),
    # S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We'll use R1 and S1 for breakout entries
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    camarilla_range = prev_high - prev_low
    r1 = prev_close + (camarilla_range * 1.1 / 12)
    s1 = prev_close - (camarilla_range * 1.1 / 12)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above R1 with volume AND above 1d EMA34 (uptrend)
            long_condition = (curr_close > r1_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below S1 with volume AND below 1d EMA34 (downtrend)
            short_condition = (curr_close < s1_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA34 or below R1 (failed breakout)
            if curr_close < ema_trend or curr_close < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA34 or above S1 (failed breakout)
            if curr_close > ema_trend or curr_close > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0