#!/usr/bin/env python3
"""
6h Williams Alligator Breakout with 12h EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence (all lines intertwined) vs presence (lines diverged, ordered).
A breakout above/below the Alligator's lips with 12h EMA50 trend alignment and volume spike (>2x 20-bar vol MA) captures strong momentum.
Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years on 6h timeframe.
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
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator on primary 6h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(src, length):
        """Smoothed Moving Average"""
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < length:
            return result
        # First value is SMA
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CLOSE) / LENGTH
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First shifted values are invalid
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculation, EMA50, and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_12h_aligned[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator alignment: check if lines are properly ordered (trending state)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        # Breakout conditions: price breaks above/below lips with alignment
        breakout_above_lips = curr_close > lips_val
        breakout_below_lips = curr_close < lips_val
        
        if position == 0:
            # Long: break above lips + bullish alignment + price above 12h EMA50 + volume confirmation
            long_signal = breakout_above_lips and bullish_alignment and (curr_close > ema_50_val) and volume_confirm
            # Short: break below lips + bearish alignment + price below 12h EMA50 + volume confirmation
            short_signal = breakout_below_lips and bearish_alignment and (curr_close < ema_50_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below lips OR alignment breaks OR price crosses below 12h EMA50
            if (curr_close < lips_val) or not bullish_alignment or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above lips OR alignment breaks OR price crosses above 12h EMA50
            if (curr_close > lips_val) or not bearish_alignment or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0