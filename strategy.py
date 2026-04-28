#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend absence (all lines intertwined = chop)
# and trend presence (lines diverged in order: Lips > Teeth > Jaw = uptrend, reverse = downtrend).
# Enter long when Alligator shows uptrend alignment + volume spike + chop < 61.8 (trending regime)
# Enter short when Alligator shows downtrend alignment + volume spike + chop < 61.8
# Exit when Alligator lines re-intertwine (chop > 61.8) or opposite alignment occurs.
# Uses 12h timeframe with 1d HTF for Alligator and chop calculation, volume confirmation on 12h.
# Designed to work in both bull/bear markets by requiring trending regime (chop filter) and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_WilliamsAlligator_ChopRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Chop calculation (requires daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:  # Need sufficient data for SMAs and chop
        return np.zeros(n)
    
    # Calculate Williams Alligator SMAs on 1d close
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Chopiness Index on 1d (to determine trending vs ranging regime)
    # Chop = 100 * log10(sum(ATR(14)) / (n * log10(n))) / log10(n)
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = strong trend
    tr = np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                               abs(df_1d['high'] - df_1d['close'].shift(1))),
                     abs(df_1d['low'] - df_1d['close'].shift(1))).values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate Chop: 100 * log10(sum(ATR14 over n periods) / (n * ATR14)) / log10(n)
    # Using common approximation: Chop = 100 * log10(sum(ATR14) / (n * ATR14)) / log10(n)
    # We'll use a rolling window of 14 for chop calculation
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    n_periods = 14
    chop = 100 * (np.log10(sum_atr_14 / (n_periods * atr_14)) / np.log10(n_periods))
    # Handle division by zero and invalid values
    chop = np.where((atr_14 > 0) & (sum_atr_14 > 0), chop, 50.0)  # Default to neutral chop
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: >1.5x 20-bar average volume on 12h
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure sufficient history for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams Alligator alignment
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        # Uptrend: Lips > Teeth > Jaw (all lines above and in order)
        alligator_uptrend = (lips_val > teeth_val) and (teeth_val > jaw_val)
        # Downtrend: Lips < Teeth < Jaw (all lines below and in order)
        alligator_downtrend = (lips_val < teeth_val) and (teeth_val < jaw_val)
        # Chop regime filter: only trade when chop < 61.8 (trending regime)
        trending_regime = chop_aligned[i] < 61.8
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator uptrend + volume confirm + trending regime
            if alligator_uptrend and vol_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator downtrend + volume confirm + trending regime
            elif alligator_downtrend and vol_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on loss of uptrend alignment or chop > 61.8 (ranging)
            if not alligator_uptrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on loss of downtrend alignment or chop > 61.8 (ranging)
            if not alligator_downtrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals