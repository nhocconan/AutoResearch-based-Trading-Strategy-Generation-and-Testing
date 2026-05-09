#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d (periods: 13, 8, 5 with shifts 8, 5, 3)
    close_1d = df_1d['close'].values
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (Green): 5-period SMMA, shifted 3 bars
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d EMA34 for trend filter (additional confirmation)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Need 34 for EMA34 and 6 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips > teeth > jaw
        alligator_short = lips < teeth < jaw
        
        if position == 0:
            # Enter long: Alligator aligned up + price > EMA34 + volume spike
            if alligator_long and close[i] > ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned down + price < EMA34 + volume spike
            elif alligator_short and close[i] < ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price < EMA34
            if not alligator_long or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price > EMA34
            if not alligator_short or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals