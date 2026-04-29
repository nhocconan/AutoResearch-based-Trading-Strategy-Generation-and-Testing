#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator identifies trending vs ranging markets via three smoothed lines (Jaw/Teeth/Lips).
# Trend signal: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish).
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend whipsaws.
# Volume spike (>1.8x 30-period average) confirms strong participation, reducing false breakouts.
# Discrete position sizing (0.25) minimizes fee churn.
# Target: 75-150 total trades over 4 years (19-37/year) on 4h timeframe.

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().values
    
    # Shift as per Alligator definition (jaw shifted most, lips least)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for lookback period
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 30-period average volume for spike confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 30)  # 1d EMA50, Alligator shifts, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_jaw = jaw_shifted[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_30[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.8x 30-period average
        vol_spike = curr_volume > 1.8 * curr_vol_ma
        
        # Williams Alligator trend conditions
        bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: bearish alignment OR price crosses below 1d EMA50
            if bearish_alignment or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish alignment OR price crosses above 1d EMA50
            if bullish_alignment or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish alignment AND above 1d EMA50 AND volume spike
            if bullish_alignment and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment AND below 1d EMA50 AND volume spike
            elif bearish_alignment and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals