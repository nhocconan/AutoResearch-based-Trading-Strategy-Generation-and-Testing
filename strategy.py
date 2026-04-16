#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + Choppiness Regime Filter
# Long when: Green (Jaw < Teeth < Lips) AND Price > Lips AND Volume Spike (2x 12h avg) AND Chop > 61.8 (ranging)
# Short when: Red (Lips < Teeth < Jaw) AND Price < Lips AND Volume Spike (2x 12h avg) AND Chop > 61.8 (ranging)
# Exit when: Chop < 38.2 (trending) OR Alligator alignment breaks
# Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trend in ranging markets
# Volume spike adds conviction to entries
# Choppiness filter (61.8 threshold) ensures we only trade in ranging markets where Alligator works best
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams Alligator (Jaw, Teeth, Lips) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMMA, 8 periods offset
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, 5 periods offset
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, 3 periods offset
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 12h Choppiness Index (14-period) ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min high/low over 14 periods
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high14 - min_low14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range14 = max_high14 - min_low14
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(sum_tr14 / range14) / np.log10(14)
    
    # === 12h Volume Spike (2x average) ===
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values  # 2 periods of 12h = 24h equivalent
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop[i]
        vol_spike = volume[i] > vol_ma[i] * 2.0  # 2x average volume
        
        # === EXIT CONDITIONS ===
        if position == 1:  # Long position
            # Exit if: chop < 38.2 (trending) OR Alligator alignment breaks (not green)
            if chop_val < 38.2 or not (jaw_val < teeth_val < lips_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if: chop < 38.2 (trending) OR Alligator alignment breaks (not red)
            if chop_val < 38.2 or not (lips_val < teeth_val < jaw_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Green Alligator: Jaw < Teeth < Lips (bullish alignment)
            is_green = jaw_val < teeth_val < lips_val
            # Red Alligator: Lips < Teeth < Jaw (bearish alignment)
            is_red = lips_val < teeth_val < jaw_val
            
            # Long when: Green AND Price > Lips AND Volume Spike AND Chop > 61.8 (ranging)
            if is_green and price > lips_val and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Red AND Price < Lips AND Volume Spike AND Chop > 61.8 (ranging)
            elif is_red and price < lips_val and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0