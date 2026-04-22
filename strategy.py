#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Alligator with 1-week EMA(34) trend filter and volume spike confirmation.
Trades Alligator crossovers in the direction of the weekly trend only when volume exceeds 2x the 20-period average.
Uses 60-minute time-based exits to limit exposure in choppy markets.
Targets 50-150 total trades over 4 years (12-37/year) with disciplined entry/exit to minimize fee drift.
Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Williams Alligator - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 40:
        return np.zeros(n)
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shifts)
    close_6h = df_6h['close'].values
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # 13,8
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # 8,5
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values   # 5,3
    
    # Apply future shifts (Alligator specific)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Zero out invalid shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Lips > Teeth > Jaw (bullish alignment) and above weekly EMA (uptrend)
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Jaws > Teeth > Lips (bearish alignment) and below weekly EMA (downtrend)
            elif jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        else:
            # Exit conditions: time-based exit (max 6 bars = 36 hours) or Alligator reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator reverses or time exit
                if lips_aligned[i] < teeth_aligned[i] or bars_since_entry >= 6:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator reverses or time exit
                if lips_aligned[i] > teeth_aligned[i] or bars_since_entry >= 6:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Alligator_1wEMA34_Volume"
timeframe = "6h"
leverage = 1.0