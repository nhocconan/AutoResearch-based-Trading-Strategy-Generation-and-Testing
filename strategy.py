#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyAlligator_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator indicator
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 34:
        return np.zeros(n)
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(source, length):
        sma = np.full_like(source, np.nan, dtype=float)
        if len(source) < length:
            return sma
        # First value is simple SMA
        sma[length-1] = np.mean(source[:length])
        # Subsequent values: (prev_sma * (length-1) + current_price) / length
        for i in range(length, len(source)):
            sma[i] = (sma[i-1] * (length-1) + source[i]) / length
        return sma
    
    # Weekly price data
    weekly_close = df_w['close'].values
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    
    # Calculate Alligator lines
    jaw_raw = smma(weekly_close, 13)  # 13-period SMMA
    teeth_raw = smma(weekly_close, 8)  # 8-period SMMA
    lips_raw = smma(weekly_close, 5)   # 5-period SMMA
    
    # Shift forward: Jaw +8, Teeth +5, Lips +3
    jaw_shifted = np.full_like(jaw_raw, np.nan)
    teeth_shifted = np.full_like(teeth_raw, np.nan)
    lips_shifted = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw_shifted[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth_shifted[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips_shifted[3:] = lips_raw[:-3]
    
    # Align to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_w, lips_shifted)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        # Alligator sleeping condition: lines are intertwined (market ranging)
        # When lips, teeth, and jaw are close together, market is sleeping
        jaw_teeth_diff = np.abs(jaw_aligned[i] - teeth_aligned[i])
        teeth_lips_diff = np.abs(teeth_aligned[i] - lips_aligned[i])
        lips_jaw_diff = np.abs(lips_aligned[i] - jaw_aligned[i])
        
        # Sleeping threshold: average difference < 0.5% of price
        avg_diff = (jaw_teeth_diff + teeth_lips_diff + lips_jaw_diff) / 3
        sleeping = avg_diff < (close[i] * 0.005)
        
        # Awakening: lines diverge and lips crosses teeth or jaw
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        lips_above_jaw = lips_aligned[i] > jaw_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        lips_below_jaw = lips_aligned[i] < jaw_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Wait for Alligator to awaken (not sleeping) and volume confirmation
            if not sleeping and vol_ok:
                # Go long when lips crosses above teeth AND jaw (bullish alignment)
                if lips_above_teeth and lips_above_jaw and teeth_above_jaw:
                    signals[i] = 0.25
                    position = 1
                # Go short when lips crosses below teeth AND jaw (bearish alignment)
                elif lips_below_teeth and lips_below_jaw and teeth_below_jaw:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long when lips crosses back below teeth (loss of bullish momentum)
            if lips_below_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when lips crosses back above teeth (loss of bearish momentum)
            if lips_above_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals