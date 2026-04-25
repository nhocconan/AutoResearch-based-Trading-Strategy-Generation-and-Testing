#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 6h identifies trend alignment, while 1d EMA50 filters for higher-timeframe direction and volume spike (>2x 20-bar MA) confirms momentum. Alligator is effective in both trending and ranging markets - in trends, the lines diverge (Lips > Teeth > Jaw for uptrend, reverse for downtrend); in ranges, they intertwine. Combined with 1d EMA50 trend filter and volume confirmation, this captures strong momentum moves while avoiding false signals in chop. Targets 50-150 total trades over 4 years to minimize fee drag. Works in bull markets via long alignments and in bear markets via short alignments.
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
    
    # Get 1d data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA, shifted 8 bars
    teeth = smma(close, 8)  # Teeth: 8-period SMMA, shifted 5 bars
    lips = smma(close, 5)   # Lips: 5-period SMMA, shifted 3 bars
    
    # Shift as per Alligator definition (Jaw 8, Teeth 5, Lips 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Invalidate shifted values
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, EMA50_1d, volume MA to propagate
    start_idx = max(50, 13+8, 20)  # EMA50_1d, Jaw shift, vol MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        curr_close = close[i]
        ema50_1d = ema_50_1d_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        if position == 0:
            # Long: Alligator bullish alignment + price above 1d EMA50 + volume confirmation
            long_condition = bullish_alignment and (curr_close > ema50_1d) and volume_confirm
            # Short: Alligator bearish alignment + price below 1d EMA50 + volume confirmation
            short_condition = bearish_alignment and (curr_close < ema50_1d) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR price crosses below 1d EMA50
            if not bullish_alignment or (curr_close < ema50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR price crosses above 1d EMA50
            if not bearish_alignment or (curr_close > ema50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0