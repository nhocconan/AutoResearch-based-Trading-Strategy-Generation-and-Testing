#!/usr/bin/env python3
"""
12h_1d_Williams_Alligator_Trend_Follow_v1
Hypothesis: Uses Williams Alligator (3 SMAs) on 12h with 1d trend filter and volume confirmation.
Designed for low trade frequency (<30/year) to avoid fee drag. Works in bull/bear via trend alignment.
Only enters when price is outside Alligator's jaws and aligned with 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Williams_Alligator_Trend_Follow_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Alligator conditions: jaws closed (all lines intertwined) vs open
        # Mouth open when lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Alligator mouth open AND aligned with 1d trend
        long_entry = (lips_above_teeth and teeth_above_jaw and 
                     price_above_1d_ema and volume_filter)
        short_entry = (lips_below_teeth and teeth_below_jaw and 
                      price_below_1d_ema and volume_filter)
        
        # Exit conditions: Alligator mouths close (lines intertwine) or trend reversal
        # Mouth close when teeth crosses lips or jaw crosses teeth
        lips_cross_teeth = (lips[i] <= teeth[i] and lips[i-1] >= teeth[i-1]) or \
                          (lips[i] >= teeth[i] and lips[i-1] <= teeth[i-1])
        teeth_cross_jaw = (teeth[i] <= jaw[i] and teeth[i-1] >= jaw[i-1]) or \
                         (teeth[i] >= jaw[i] and teeth[i-1] <= jaw[i-1])
        mouth_close = lips_cross_teeth or teeth_cross_jaw
        
        long_exit = mouth_close or (not price_above_1d_ema)
        short_exit = mouth_close or (not price_below_1d_ema)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals