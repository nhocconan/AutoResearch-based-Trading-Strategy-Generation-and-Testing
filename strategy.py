#!/usr/bin/env python3
"""
12h_Williams_Alligator_1wTrend_VolumeConfirm
Hypothesis: Williams Alligator on 12h with 1w trend filter and volume confirmation. Works in bull/bear markets by using Alligator jaws-teeth-lips for trend direction and separation strength, filtered by 1w EMA50 and volume spikes. Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25) to minimize fee drag.
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Williams Alligator on 12h: SMAs of median price
    # Jaw: 13-period SMA, 8-bar shift
    # Teeth: 8-period SMA, 5-bar shift  
    # Lips: 5-period SMA, 3-bar shift
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 13, 8, 5, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Alligator separation strength (avoid choppy markets)
        jaw_teeth_sep = np.abs(teeth[i] - jaw[i]) / (at[i] + 1e-10)
        teeth_lips_sep = np.abs(lips[i] - teeth[i]) / (atr[i] + 1e-10)
        separation_ok = (jaw_teeth_sep > 0.5) and (teeth_lips_sep > 0.3)
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: Alligator uptrend + price > 1w EMA50 (uptrend filter) + volume spike + separation
        if alligator_long and close[i] > ema_50_1w_aligned[i] and volume_spike[i] and separation_ok:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Alligator downtrend + price < 1w EMA50 (downtrend filter) + volume spike + separation
        elif alligator_short and close[i] < ema_50_1w_aligned[i] and volume_spike[i] and separation_ok:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # ATR-based stoploss: exit if price moves against position by 2.5 * ATR
        elif position == 1 and close[i] < lips[i] - 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > lips[i] + 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        # Exit Alligator signal: lips crosses teeth in opposite direction
        elif position == 1 and lips[i] < teeth[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and lips[i] > teeth[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Williams_Alligator_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0