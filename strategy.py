#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) crossover with 1d volume confirmation and 1w ADX > 20 trend filter.
# Long when Alligator Lips cross above Teeth AND volume > 1.3x 20-bar average AND 1w ADX > 20.
# Short when Alligator Lips cross below Teeth AND volume > 1.3x 20-bar average AND 1w ADX > 20.
# Exit when Lips cross back over Teeth (reverse crossover).
# Uses discrete position size 0.25. Alligator identifies trend direction and entry timing,
# volume confirms breakout strength, 1w ADX ensures we trade only in trending regimes (avoids chop).
# Target: 40-100 trades over 4 years (10-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams Alligator (13,8,5) ===
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === 1d Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data once before loop for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max shift is 8 for Jaw)
    warmup = 15
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        vol_ma_val = vol_ma_20[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.3x 20-period average
        vol_filter = vol > 1.3 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1w ADX > 20 (trending regime)
        trend_filter = adx_val > 20
        
        # === Crossover Detection ===
        # Lips above Teeth (bullish alignment)
        lips_above_teeth = lips_val > teeth_val
        # Lips below Teeth (bearish alignment)
        lips_below_teeth = lips_val < teeth_val
        
        # Previous values for crossover detection
        if i > 0:
            prev_lips_above_teeth = lips[i-1] > teeth[i-1]
            prev_lips_below_teeth = lips[i-1] < teeth[i-1]
        else:
            prev_lips_above_teeth = False
            prev_lips_below_teeth = False
        
        # Bullish crossover: Lips cross above Teeth
        bullish_cross = lips_above_teeth and not prev_lips_above_teeth
        # Bearish crossover: Lips cross below Teeth
        bearish_cross = lips_below_teeth and not prev_lips_below_teeth
        
        # === EXIT LOGIC (reverse crossover) ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Lips cross back below Teeth
            if lips_below_teeth and prev_lips_above_teeth:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Lips cross back above Teeth
            if lips_above_teeth and prev_lips_below_teeth:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Lips cross above Teeth with volume and trend confirmation
            if bullish_cross and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Lips cross below Teeth with volume and trend confirmation
            elif bearish_cross and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Alligator_1dVolumeSpike_1wADX20Trend_V1"
timeframe = "1d"
leverage = 1.0