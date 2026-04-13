#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray power + volume confirmation
# Williams Alligator identifies trend direction via smoothed SMAs (Jaw/Teeth/Lips)
# Elder Ray measures bull/bear power relative to EMA13
# Volume confirmation ensures momentum behind moves
# Works in bull (Alligator aligned up, bull power > 0) and bear (aligned down, bear power < 0)
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5 smoothed by 8,5,3 periods)
    # Jaw: SMA(13, 8) - blue line
    # Teeth: SMA(8, 5) - red line  
    # Lips: SMA(5, 3) - green line
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray Power (requires EMA13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Average volume (20-period) for confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Alligator aligned up (Lips > Teeth > Jaw) + Bull Power > 0 + Volume
            if (lips_val > teeth_val and teeth_val > jaw_val and 
                bull_val > 0 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Alligator aligned down (Lips < Teeth < Jaw) + Bear Power < 0 + Volume
            elif (lips_val < teeth_val and teeth_val < jaw_val and 
                  bear_val < 0 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not (lips_val > teeth_val and teeth_val > jaw_val and bull_val > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not (lips_val < teeth_val and teeth_val < jaw_val and bear_val < 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0