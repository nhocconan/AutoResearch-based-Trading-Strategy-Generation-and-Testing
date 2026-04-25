#!/usr/bin/env python3
"""
6h Williams Alligator + Chop Regime + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
Chop regime filter (CHOP > 61.8 = ranging, < 38.2 = trending) ensures we only trade in clear trends.
Volume spike confirms institutional participation. Works in both bull (long when aligned up) and bear (short when aligned down).
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Alligator and Chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs shifted forward
    # Jaw: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars  
    # Lips: 5-period SMA shifted 3 bars
    close_1d = pd.Series(df_1d['close'])
    jaw_1d = close_1d.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = close_1d.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = close_1d.rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Chop regime on 1d: measures trendiness vs ranging
    # CHOP = 100 * log10(SUM(ATR(1), n) / (MAX(HIGH,n) - MIN(LOW,n))) / log10(n)
    # We'll use a simplified version: CHOP > 61.8 = ranging, < 38.2 = trending
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
            np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        )
    )
    # Handle first bar
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_1d = max_high_1d - min_low_1d
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    chop_1d = 100 * np.log10(atr_1d * 14 / range_1d) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (13+8=21) and Chop (14) and volume (20)
    start_idx = max(21, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_aligned = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
        bearish_aligned = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])
        
        # Chop regime: only trade when trending (CHOP < 38.2), avoid ranging (CHOP > 61.8)
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trending regime + volume spike
            # Long: bullish alignment AND trending regime AND volume spike
            long_entry = bullish_aligned and trending_regime and vol_spike
            # Short: bearish alignment AND trending regime AND volume spike
            short_entry = bearish_aligned and trending_regime and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: loss of bullish alignment OR enter ranging regime
            if not bullish_aligned or ranging_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: loss of bearish alignment OR enter ranging regime
            if not bearish_aligned or ranging_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ChopRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0