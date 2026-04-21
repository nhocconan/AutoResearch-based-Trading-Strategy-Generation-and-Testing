#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and chop regime filter
# Long when Jaw < Teeth < Lips (bullish alignment) and price > Lips, with volume > 1.5x 20-day average and CHOP > 50
# Short when Jaw > Teeth > Lips (bearish alignment) and price < Lips, with volume > 1.5x 20-day average and CHOP > 50
# Williams Alligator uses SMAs: Jaw=13, Teeth=8, Lips=5 (all shifted forward)
# Works in trending markets (CHOP < 50 indicates trend, but we use CHOP > 50 for ranging markets to fade false breaks)
# Actually, we want CHOP < 50 for trending markets. Let me correct: CHOP < 50 = trend, CHOP > 50 = range
# We'll use CHOP < 50 to identify trending markets for Alligator signals
# Volume confirms conviction, chop filter ensures we're in a trending environment
# Target: 20-35 trades/year by requiring multiple confirmations

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components on 1d
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs, then shifted
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift the lines (Williams Alligator shifts jaw by 8, teeth by 5, lips by 3)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d (to identify trending vs ranging markets)
    # CHOP = 100 * log10(sum(ATR over n periods) / (max(high) - min(low))) / log10(n)
    # We'll use 14-period CHOP
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                abs(high_1d[i] - close_1d[i-1]), 
                abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr
    
    # Smooth ATR with 14-period average
    atr_ma_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP: 100 * log10(sum(ATR14 over 14 periods) / (max(high14) - min(low14))) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_ma_1d[i-13:i+1])  # 14 periods of ATR
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low:  # avoid division by zero
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align all 1d indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-day average (scaled)
        # Since we're comparing 4h volume to daily MA, we need to scale
        # Approximate: 4h volume should be > 1.5 * (daily_vol_ma / 6) since 6x 4h bars per day
        volume_confirm = volume > 1.5 * (vol_ma / 6.0)
        
        # Chop filter: CHOP < 50 indicates trending market (good for Alligator)
        chop_filter = chop_val < 50
        
        if position == 0:
            # Long: Bullish alignment (Jaw < Teeth < Lips) and price > Lips
            if jaw_val < teeth_val < lips_val and price > lips_val and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Jaw > Teeth > Lips) and price < Lips
            elif jaw_val > teeth_val > lips_val and price < lips_val and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if bearish alignment forms or price crosses below Teeth
                if jaw_val > teeth_val > lips_val or price < teeth_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if bullish alignment forms or price crosses above Teeth
                if jaw_val < teeth_val < lips_val or price > teeth_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolume_ChopFilter"
timeframe = "4h"
leverage = 1.0