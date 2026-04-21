#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Fractal Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength. 
Trades trigger when price crosses all three lines with volume confirmation, using weekly 
fractals for trend validation. Works in both bull/bear markets by capturing strong 
trends while avoiding whipsaws via fractal confirmation. Designed for 12h timeframe 
with low trade frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for fractals (higher timeframe trend validation)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on weekly data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_weekly['high'].values,
        df_weekly['low'].values,
    )
    # Add 2-bar delay for fractal confirmation (needs 2 future weekly bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_weekly, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_weekly, bullish_fractal, additional_delay_bars=2
    )
    
    # Load daily data for Alligator components (smoothed medians)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 13:
        return np.zeros(n)
    
    median_price = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3.0
    median_values = median_price.values
    
    # Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    jaw = pd.Series(median_values).rolling(window=13, min_periods=13).median().shift(8).values
    teeth = pd.Series(median_values).rolling(window=8, min_periods=8).median().shift(5).values
    lips = pd.Series(median_values).rolling(window=5, min_periods=5).median().shift(3).values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bearish_fract = bearish_fractal_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.8x 24-period average (balanced frequency)
        if i >= 24:
            vol_ma = np.mean(volume[i-24:i])
        else:
            vol_ma = volume[i]
        vol_ok = vol_current > 1.8 * vol_ma
        
        # Alligator alignment checks
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + bullish fractal + volume
            if lips_above_teeth and teeth_above_jaw and bullish_fract and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) + bearish fractal + volume
            elif lips_below_teeth and teeth_below_jaw and bearish_fract and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips crosses below Teeth OR bearish fractal appears
            if lips_val < teeth_val or bearish_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips crosses above Teeth OR bullish fractal appears
            if lips_val > teeth_val or bullish_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Fractal_Volume"
timeframe = "12h"
leverage = 1.0