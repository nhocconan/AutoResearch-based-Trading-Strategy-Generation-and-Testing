#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike Confirmation
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence (all lines intertwined) vs presence (diverged). In 12h timeframe, we trade only when Alligator is "awake" (Lips > Teeth > Jaw for uptrend, reverse for downtrend) AND price is beyond the outer lines, with 1d EMA50 trend filter and volume confirmation (>2x 20-bar vol MA). This avoids choppy markets and captures strong trends. Designed for low trade frequency (<30/year) to minimize fee drag while working in both bull (buy strength) and bear (sell weakness) regimes.
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
    
    # Get 12h data for Williams Alligator (SMMA-based)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for SMMA periods
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: SMMA(close, 13), SMMA(high, 8), SMMA(low, 5) with shifts
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Smoothed Moving Average (SMMA) = EMA-like but with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        # First value: simple average
        res[period-1] = np.mean(arr[:period])
        # Subsequent: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    # Alligator lines: Jaw (blue, 13-period SMMA of median price, shifted 8 bars)
    #              Teeth(red, 8-period SMMA of median price, shifted 5 bars)
    #              Lips(green,5-period SMMA of median price, shifted 3 bars)
    median_12h = (high_12h + low_12h) / 2
    jaw = smma(median_12h, 13)  # 13-period
    teeth = smma(median_12h, 8)  # 8-period
    lips = smma(median_12h, 5)   # 5-period
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Invalidate shifted entries
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align to 12h timeframe (already 12h, but use align_htf_to_ltf for safety with potential gaps)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (50), EMA50 (51), volume MA (20)
    start_idx = max(50, 51, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator "awake" conditions: lines diverged in order
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        alligator_up = (lips_val > teeth_val) and (teeth_val > jaw_val)
        alligator_down = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Price position relative to Alligator: beyond outer lines
        price_above_lips = curr_close > lips_val
        price_below_lips = curr_close < lips_val
        
        if position == 0:
            # Long: Alligator awake uptrend + price above Lips + above 1d EMA50 + volume confirmation
            long_signal = alligator_up and price_above_lips and (curr_close > ema_50_val) and volume_confirm
            # Short: Alligator awake downtrend + price below Lips + below 1d EMA50 + volume confirmation
            short_signal = alligator_down and price_below_lips and (curr_close < ema_50_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns (Lips crosses below Teeth) OR price crosses below Jaw
            if (lips_val < teeth_val) or (curr_close < jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns (Lips crosses above Teeth) OR price crosses above Jaw
            if (lips_val > teeth_val) or (curr_close > jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0