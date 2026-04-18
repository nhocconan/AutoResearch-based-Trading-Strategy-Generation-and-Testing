#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + ADX Trend
Hypothesis: Williams Alligator (three SMAs) identifies trend alignment and momentum.
Combined with volume spike (confirming institutional interest) and ADX > 25 (trending regime),
this strategy captures strong trending moves on 12h timeframe. Works in both bull and bear
markets by only trading in the direction of strong trends, avoiding chop. Targets 15-25
trades/year to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d data for ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_1d_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d_safe
    
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # Long: Alligator aligned up, ADX > 25, volume spike
            if lips_val > teeth_val and teeth_val > jaw_val and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down, ADX > 25, volume spike
            elif lips_val < teeth_val and teeth_val < jaw_val and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if Alligator alignment breaks or ADX weakens
            if not (lips_val > teeth_val and teeth_val > jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if Alligator alignment breaks or ADX weakens
            if not (lips_val < teeth_val and teeth_val < jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0