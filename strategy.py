#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy with 1w Williams Alligator trend filter and 1d Bollinger Bands reversal
# Williams Alligator uses three SMAs (Jaw, Teeth, Lips) to identify trend direction and strength
# Bollinger Bands %B identifies overbought/oversold conditions for mean reversion entries
# Trend filter prevents counter-trend trades in strong moves, while Bollinger Bands capture mean reversion within the trend
# Weekly timeframe reduces noise and improves signal quality for daily signals
# Designed to work in both bull and bear markets by aligning with higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(df_1w['close']).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(df_1w['close']).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(df_1w['close']).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_vals)
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_len = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    
    # Basis (SMA)
    basis = pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).mean().values
    # Deviation
    dev = bb_mult * pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).std().values
    # Upper and Lower bands
    upper = basis + dev
    lower = basis - dev
    # Percent B (%B)
    bb_pctb = (bb_src - lower) / (upper - lower)
    bb_pctb = np.where((upper - lower) == 0, 0.5, bb_pctb)  # Avoid division by zero
    
    # Align BB %B to 1d timeframe
    bb_pctb_aligned = align_htf_to_ltf(prices, df_1d, bb_pctb)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 21)  # 21 for Alligator (13+8 shift)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(bb_pctb_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: Alligator alignment indicates trend
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Mean reversion signals from Bollinger Bands %B
        oversold = bb_pctb_aligned[i] < 0.2
        overbought = bb_pctb_aligned[i] > 0.8
        
        if position == 0:
            # Enter long: bullish trend + oversold (pullback in uptrend)
            if bullish_alignment and oversold:
                position = 1
                signals[i] = position_size
            # Enter short: bearish trend + overbought (pullback in downtrend)
            elif bearish_alignment and overbought:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above 50% BB level OR Alligator alignment breaks
            if bb_pctb_aligned[i] > 0.5 or not bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below 50% BB level OR Alligator alignment breaks
            if bb_pctb_aligned[i] < 0.5 or not bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wWilliamsAlligator_1dBB_PB_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0