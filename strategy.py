#!/usr/bin/env python3
"""
1d Weekly Pivot R1S1 Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Weekly Camarilla pivot levels (R1/S1) act as strong support/resistance on the daily timeframe. 
Breakouts above R1 or below S1 with volume confirmation and aligned 1w EMA34 trend capture institutional moves 
while reducing false breakouts. Designed for very low trade frequency (7-25/year) on 1d timeframe to work in 
both bull and bear markets via trend following with strict entry conditions.
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
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1w data for Camarilla pivot calculation (R1, S1 levels)
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 1w bar: based on previous week's high, low, close
    # We need to shift to avoid look-ahead: use previous week's data to calculate current week's levels
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla formulas for R1/S1 (tighter levels than R3/S3 for more precision):
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to LTF (1d) - no extra delay needed as pivots are based on completed 1w bar
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Calculate volume spike: current volume > 2.5 * 20-period average volume (stricter for 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and to avoid NaN from shift
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 resistance AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > r1_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S1 support AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < s1_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below S1 support (broken support) OR price crosses below EMA (trend change)
            if (curr_close < s1_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above R1 resistance (broken resistance) OR price crosses above EMA (trend change)
            if (curr_close > r1_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0