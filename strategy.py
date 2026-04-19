# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_GatorOsc_Trend_HTF1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Alligator (Williams' Alligator) on daily close
    # Jaw (blue): 13-period SMMA, smoothed 8 bars ahead
    # Teeth (red): 8-period SMMA, smoothed 5 bars ahead  
    # Lips (green): 5-period SMMA, smoothed 3 bars ahead
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(values, period):
        sma = np.full_like(values, np.nan, dtype=np.float64)
        smma_vals = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) >= period:
            # First value is SMA
            sma[period-1] = np.mean(values[:period])
            smma_vals[period-1] = sma[period-1]
            # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
            for i in range(period, len(values)):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + values[i]) / period
        return smma_vals
    
    # Calculate SMMA series
    close_1d = df_1d['close'].values
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply smoothing (shift forward)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Gator Oscillator: |Jaw-Teeth| and |Teeth-Lips| (absolute values)
    jaw_teeth_diff = np.abs(jaw_aligned - teeth_aligned)
    teeth_lips_diff = np.abs(teeth_aligned - lips_aligned)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        
        # Alligator alignment check: all three lines in proper order
        # For uptrend: Lips > Teeth > Jaw (green > red > blue)
        # For downtrend: Jaw > Teeth > Lips (blue > red > green)
        vol_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume
            if lips > teeth and teeth > jaw and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + volume
            elif jaw > teeth and teeth > lips and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Lips cross below Teeth or volume confirmation lost
            if lips < teeth or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Jaw crosses below Teeth or volume confirmation lost
            if jaw < teeth or not vol_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals