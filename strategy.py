#!/usr/bin/env python3
"""
1d Williams Alligator + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend strength and direction on daily timeframe; 1w EMA50 ensures alignment with weekly trend; volume spike confirms conviction. Designed for 1d timeframe to target 7-25 trades/year (30-100 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the weekly trend and avoiding counter-trend entries.
"""

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
    
    # Load weekly data ONCE before loop for indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on daily (using prices directly)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(src, length):
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < length:
            return result
        # First value is simple SMA
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations (max shift is 8 for jaw)
    start_idx = max(13, 8, 5, 20, 50) + 8  # +8 for jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator trend detection: 
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        # Bearish: Jaw > Teeth > Lips (all aligned downward)
        bullish_aligned = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_aligned = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        
        # Weekly trend filter: price relative to weekly EMA50
        weekly_bullish = curr_close > ema_50_1w_aligned[i]
        weekly_bearish = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator bullish AND weekly bullish AND volume spike
            long_entry = bullish_aligned and weekly_bullish and vol_spike
            # Short: Alligator bearish AND weekly bearish AND volume spike
            short_entry = bearish_aligned and weekly_bearish and vol_spike
            
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
            # Exit: Alligator loses bullish alignment OR loses weekly bullish bias
            if not (bullish_aligned and weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses bearish alignment OR loses weekly bearish bias
            if not (bearish_aligned and weekly_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0