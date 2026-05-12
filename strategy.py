#!/usr/bin/env python3
name = "6h_Alligator_ElderRay_DualFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator aligned: Jaw > Teeth > Lips = bullish, reverse = bearish
    alligator_bull = (jaw > teeth) & (teeth > lips)
    alligator_bear = (jaw < teeth) & (teeth < lips)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 + Bear Power < 0 + Alligator bullish + volume
            if (bull_power_6h[i] > 0) and (bear_power_6h[i] < 0) and alligator_bull[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + Bull Power < 0 + Alligator bearish + volume
            elif (bear_power_6h[i] < 0) and (bull_power_6h[i] < 0) and alligator_bear[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power turns negative
            if (not alligator_bull[i]) or (bull_power_6h[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power turns positive
            if (not alligator_bear[i]) or (bear_power_6h[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals