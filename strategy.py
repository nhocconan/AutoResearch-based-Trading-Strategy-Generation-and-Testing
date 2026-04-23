#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) crossover with 1d EMA50 trend filter and volume confirmation.
Long when Alligator Lips cross above Teeth AND 1d EMA50 rising AND 12h volume > 1.5x 20-period MA.
Short when Alligator Lips cross below Teeth AND 1d EMA50 falling AND 12h volume > 1.5x 20-period MA.
Exit when Lips cross back opposite direction or 1d EMA50 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator captures trend emergence, 1d EMA50 filters major trend, volume avoids false breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate Williams Alligator (12h timeframe)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (Blue line): 13-period SMMA shifted 8 bars
    def smma(source, period):
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > jaw_shift:
        jaw_shifted[jaw_shift:] = jaw[:-jaw_shift]
    if len(teeth) > teeth_shift:
        teeth_shifted[teeth_shift:] = teeth[:-teeth_shift]
    if len(lips) > lips_shift:
        lips_shifted[lips_shift:] = lips[:-lips_shift]
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 50, 20)  # Alligator shifts, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Alligator crossover
        if i >= start_idx + 1:
            lips_prev = lips_shifted[i-1]
            teeth_prev = teeth_shifted[i-1]
            lips_above_teeth = lips_val > teeth_val
            lips_below_teeth = lips_val < teeth_val
            lips_crossed_above = lips_prev <= teeth_prev and lips_above_teeth
            lips_crossed_below = lips_prev >= teeth_prev and lips_below_teeth
        else:
            lips_crossed_above = False
            lips_crossed_below = False
            lips_above_teeth = lips_val > teeth_val
            lips_below_teeth = lips_val < teeth_val
        
        # Calculate EMA50 slope for trend direction
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Lips cross above Teeth AND EMA50 rising AND volume filter
            if lips_crossed_above and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below Teeth AND EMA50 falling AND volume filter
            elif lips_crossed_below and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Lips cross back below Teeth OR EMA50 starts falling
                if lips_crossed_below or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Lips cross back above Teeth OR EMA50 starts rising
                if lips_crossed_above or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_Crossover_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0