#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike + 1w EMA50 Trend Filter
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trendless markets when lines are intertwined.
Trades only when Alligator is 'awake' (jaws separated) with volume confirmation and aligned with weekly EMA50 trend.
Works in bull markets via trend continuation and in bear markets via trend filter (avoids counter-trend entries).
Target: 10-25 trades/year on 1d to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift jaw by 8, teeth by 5, lips by 3 (Alligator smoothing)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculation and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator is 'awake' when lines are separated (not intertwined)
        # Long condition: lips > teeth > jaw (bullish alignment) AND close > lips AND volume spike AND close > weekly EMA50
        # Short condition: jaw > teeth > lips (bearish alignment) AND close < jaw AND volume spike AND close < weekly EMA50
        if position == 0:
            # Look for entry signals
            bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
            bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
            
            long_entry = bullish_alignment and (curr_close > lips_val) and vol_spike and (curr_close > ema_trend)
            short_entry = bearish_alignment and (curr_close < jaw_val) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: Alligator goes to sleep (lines intertwine) OR close < teeth (trend weakening) OR close < weekly EMA50
            lips_teeth_cross = lips_val <= teeth_val
            teeth_jaw_cross = teeth_val <= jaw_val
            alligator_sleeping = lips_teeth_cross or teeth_jaw_cross
            
            if alligator_sleeping or (curr_close < teeth_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator goes to sleep (lines intertwine) OR close > teeth (trend weakening) OR close > weekly EMA50
            lips_teeth_cross = lips_val >= teeth_val
            teeth_jaw_cross = teeth_val >= jaw_val
            alligator_sleeping = lips_teeth_cross or teeth_jaw_cross
            
            if alligator_sleeping or (curr_close > teeth_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0