#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Alligator with volume confirmation
# - Uses 1d Elder Ray (Bull/Bear Power) for trend direction from higher timeframe
# - Uses 6h Alligator (JAW/TEETH/LIPS) for entry timing and trend strength
# - Volume confirmation filters weak signals
# - Designed to work in both bull and bear markets by following the 1d trend
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_elder_ray_alligator_volume_v2"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 6h Alligator (JAW=13, TEETH=8, LIPS=5 SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # 1d Elder Ray trend direction
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # 6h Alligator alignment
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Alligator conditions
        # Alligator sleeping: all lines intertwined (no clear trend)
        sleeping = (abs(jaw_val - teeth_val) < 0.001 * close_price and 
                   abs(teeth_val - lips_val) < 0.001 * close_price and
                   abs(jaw_val - lips_val) < 0.001 * close_price)
        
        # Alligator awakening: Lips crosses Teeth with Jaw direction
        lips_above_teeth = lips_val > teeth_val
        lips_below_teeth = lips_val < teeth_val
        jaw_direction_up = jaw_val > teeth_val  # Jaw above Teeth = bullish alignment
        jaw_direction_down = jaw_val < teeth_val  # Jaw below Teeth = bearish alignment
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: 1d Bull Power positive + Lips crosses above Teeth + Jaw bullish + volume
        if (bull_power > 0 and lips_above_teeth and jaw_direction_up and 
            not sleeping and vol_confirm):
            enter_long = True
        
        # Short: 1d Bear Power negative + Lips crosses below Teeth + Jaw bearish + volume
        if (bear_power < 0 and lips_below_teeth and jaw_direction_down and 
            not sleeping and vol_confirm):
            enter_short = True
        
        # Exit conditions: opposite Alligator cross or Elder Ray divergence
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Lips crosses below Teeth or Bear Power becomes positive
            exit_long = (lips_below_teeth and teeth_val > lips_val) or (bear_power > 0)
        elif position == -1:
            # Exit short if Lips crosses above Teeth or Bull Power becomes negative
            exit_short = (lips_above_teeth and teeth_val < lips_val) or (bull_power < 0)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals