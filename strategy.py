#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + volume spike + 1d EMA50 trend filter
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to detect trends
# Long when Lips > Teeth > Jaw (bullish alignment) + volume spike + price > 1d EMA50
# Short when Lips < Teeth < Jaw (bearish alignment) + volume spike + price < 1d EMA50
# Exit when Alligator alignment breaks or volume dries up
# Target: 15-30 trades/year to avoid fee drag, works in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 12h data for Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Williams Alligator SMAs
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Align to main timeframe (12h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume spike filter (20-period average on 12h timeframe)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
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
        ema50 = ema50_aligned[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: bullish alignment + volume spike + price > EMA50
            if bullish_alignment and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + volume spike + price < EMA50
            elif bearish_alignment and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator alignment breaks or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or volume dries up
                if not bullish_alignment or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or volume dries up
                if not bearish_alignment or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_EMA50"
timeframe = "12h"
leverage = 1.0