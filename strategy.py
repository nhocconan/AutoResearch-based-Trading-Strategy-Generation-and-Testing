#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter with volume confirmation.
# The Alligator (Jaw/Teeth/Lips) identifies trend absence when lines are intertwined.
# We trade only when the Alligator is "awake" (lines separated) AND aligned with 1d EMA34.
# Volume > 1.5x 20-period average confirms momentum.
# Works in bull/bear: avoids whipsaws in ranging markets, catches strong trends.
# Target: 15-35 trades/year per symbol.
name = "6h_Alligator_EMA34_Volume_Filter"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams Alligator components (13,8,5 periods SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line (8)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line (5)
    
    # Align 1d EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6s average volume for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34, 20)  # Ensure Alligator jaw, EMA34, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_34_val = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Alligator "awake" condition: lines are separated (not intertwined)
        # For uptrend: Lips > Teeth > Jaw
        # For downtrend: Lips < Teeth < Jaw
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        is_uptrend_aligned = lips_above_teeth and teeth_above_jaw and (price > ema_34_val)
        is_downtrend_aligned = lips_below_teeth and teeth_below_jaw and (price < ema_34_val)
        
        if position == 0:
            # Look for entry when Alligator is awake and aligned with daily trend
            if is_uptrend_aligned and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif is_downtrend_aligned and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Alligator starts to sleep (lines intertwine) or trend breaks
            # Sleep condition: jaws, teeth, lips are intertwined (not separated)
            lips_teeth_crossed = abs(lips_val - teeth_val) < 0.0001 * price  # Very close
            teeth_jaw_crossed = abs(teeth_val - jaw_val) < 0.0001 * price
            is_sleeping = lips_teeth_crossed or teeth_jaw_crossed
            
            if is_sleeping or (price < ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Alligator starts to sleep or trend breaks
            lips_teeth_crossed = abs(lips_val - teeth_val) < 0.0001 * price
            teeth_jaw_crossed = abs(teeth_val - jaw_val) < 0.0001 * price
            is_sleeping = lips_teeth_crossed or teeth_jaw_crossed
            
            if is_sleeping or (price > ema_34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals