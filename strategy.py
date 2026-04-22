#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume spike confirmation.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs with forward shift) identifies trends via alignment.
# Jaw (blue): 13-period SMA shifted 8 bars forward
# Teeth (red): 8-period SMA shifted 5 bars forward  
# Lips (green): 5-period SMA shifted 3 bars forward
# When Lips > Teeth > Jaw = bullish alignment, Lips < Teeth < Jaw = bearish alignment.
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed for low trade frequency (~15-30/year on 12h) to minimize fee drift while capturing strong trends.
# Works in bull markets via bullish alignment and bears via bearish alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe (waits for 1d bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (5-period SMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator alignment signals
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bearish alignment occurs or trend breaks
                if bearish_alignment or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bullish alignment occurs or trend breaks
                if bullish_alignment or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0