#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ADX + Volume Spike
# Long when price > Alligator Jaw, ADX > 25 (trending), Volume > 1.5x MA(20)
# Short when price < Alligator Jaw, ADX > 25, Volume > 1.5x MA(20)
# Alligator identifies trend direction, ADX filters for strong trends, Volume confirms conviction
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_1dADX_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range
    tr1 = d_high - d_low
    tr2 = np.abs(d_high - np.roll(d_close, 1))
    tr3 = np.abs(d_low - np.roll(d_close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = d_high - np.roll(d_high, 1)
    down_move = np.roll(d_low, 1) - d_low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 12h
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift by half the period
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        alligator_up = lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]
        alligator_down = lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i]
        
        adx_val = adx_aligned[i]
        vol_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: Alligator up, ADX > 25, Volume confirmation
            if alligator_up and adx_val > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator down, ADX > 25, Volume confirmation
            elif alligator_down and adx_val > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator down or ADX < 20
            if not alligator_up or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator up or ADX < 20
            if not alligator_down or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals