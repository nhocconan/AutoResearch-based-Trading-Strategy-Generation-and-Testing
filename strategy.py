#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
# Williams Alligator uses three SMAs (Jaw: 13-period, Teeth: 8-period, Lips: 5-period) to identify trends.
# When the Alligator is "awake" (lines intertwined and moving in same direction) with price outside the mouth,
# it indicates a strong trend. Combined with 1d EMA50 trend filter and volume spikes (>2x 20-period average),
# this captures institutional moves while avoiding chop. Designed for low trade frequency (~15-25/year)
# to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (SMAs)
    # Jaw (blue line): 13-period SMA, 8 bars ahead
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    # Teeth (red line): 8-period SMA, 5 bars ahead  
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    # Lips (green line): 5-period SMA, 3 bars ahead
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe (waits for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
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
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator conditions: "awake" and trending
        # Awake: jaws, teeth, lips are intertwined and separated
        # Bullish: lips > teeth > jaw (green above red above blue)
        # Bearish: lips < teeth < jaw (green below red below blue)
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish Alligator + price above teeth + uptrend + volume spike
            if bullish_alligator and price > teeth_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator + price below teeth + downtrend + volume spike
            elif bearish_alligator and price < teeth_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Alligator turns bearish or price breaks below jaw or trend breaks
                if not bullish_alligator or price < jaw_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Alligator turns bullish or price breaks above jaw or trend breaks
                if not bearish_alligator or price > jaw_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0