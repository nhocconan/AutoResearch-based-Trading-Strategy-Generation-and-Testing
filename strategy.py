#!/usr/bin/env python3
name = "1d_WilliamsAlligator_JawTeeth_1wTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Williams Alligator components on 1w
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align to 1d
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # 1w trend: EMA 34
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: 20-period SMA > 0
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw AND price > EMA34 AND volume > SMA
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_sma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw AND price < EMA34 AND volume > SMA
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_sma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips <= Teeth OR Teeth <= Jaw OR price < EMA34
            if (lips_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= jaw_aligned[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Lips >= Teeth OR Teeth >= Jaw OR price > EMA34
            if (lips_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= jaw_aligned[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals