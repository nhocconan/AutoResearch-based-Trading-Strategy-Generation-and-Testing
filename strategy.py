#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw, Teeth, Lips) representing different timeframes.
# When the three lines are intertwined (no clear separation), the market is "sleeping" (range-bound).
# When they diverge in proper order (Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend),
# the market is trending and the Alligator is "awake" and "eating".
# We enter long when Alligator signals uptrend with price above Jaw, volume spike, and 1d uptrend.
# We enter short when Alligator signals downtrend with price below Jaw, volume spike, and 1d downtrend.
# This strategy works in both bull and bear markets by following the 1d trend direction.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (all based on median price)
    # Median price = (high + low) / 2
    median_price = (high_1d + low_1d) / 2
    
    # Jaw (blue line) - 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth (red line) - 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips (green line) - 5-period SMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe (waits for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator conditions:
        # Uptrend: Lips > Teeth > Jaw (green above red above blue)
        # Downtrend: Lips < Teeth < Jaw (green below red below blue)
        alligator_uptrend = lips_val > teeth_val and teeth_val > jaw_val
        alligator_downtrend = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: Alligator uptrend + price above Jaw + 1d uptrend + volume spike
            if alligator_uptrend and price > jaw_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator downtrend + price below Jaw + 1d downtrend + volume spike
            elif alligator_downtrend and price < jaw_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Alligator signals downtrend or price breaks below Jaw
                if alligator_downtrend or price < jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Alligator signals uptrend or price breaks above Jaw
                if alligator_uptrend or price > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0