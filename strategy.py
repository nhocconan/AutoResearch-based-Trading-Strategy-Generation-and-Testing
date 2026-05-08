#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation
# Long when Lips > Teeth > Jaw (bullish alignment), 1d EMA50 rising, volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment), 1d EMA50 falling, volume > 1.5x average
# Uses Alligator for trend alignment, EMA50 for trend filter, volume for confirmation
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h data
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Jaw (blue line): 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (red line): 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (green line): 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe (already aligned, but using helper for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need at least 13 periods for Alligator
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: bullish alignment (Lips > Teeth > Jaw), 1d uptrend, volume confirmation
            if lips_val > teeth_val and teeth_val > jaw_val and ema50_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (Lips < Teeth < Jaw), 1d downtrend, volume confirmation
            elif lips_val < teeth_val and teeth_val < jaw_val and ema50_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or 1d trend down
            if lips_val < teeth_val or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or 1d trend up
            if lips_val > teeth_val or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals