#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phases on 6h.
When Lips cross above Teeth/Jaw with 1d EMA34 uptrend and volume confirmation,
it captures strong bullish momentum. Reverse for bearish. Works in bull/bear
by following 1d trend while using Alligator for precise entry/exit.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Williams Alligator on 6h: SMAs with specific offsets
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8) + EMA34 + VolMA20
    start_idx = max(20, 34)  # covers shifts and indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_level = ema_34_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Exit conditions: Alligator loses alignment or trend change
        if position != 0:
            if position == 1 and (not bullish_alligator or curr_close < ema_34_level):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (not bearish_alligator or curr_close > ema_34_level):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Alligator alignment + trend + volume
        if position == 0:
            long_condition = bullish_alligator and (curr_close > ema_34_level) and volume_spike
            short_condition = bearish_alligator and (curr_close < ema_34_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0