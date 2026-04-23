#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d Supertrend trend filter and volume confirmation.
- Uses tighter H3/L3 levels for more frequent but still selective breakouts
- Supertrend(10,3) on 1d for robust trend following in both bull and bear markets
- Volume > 1.5x 20-period average for conviction (balanced to avoid overtrading)
- Long: price > H3 + volume confirmation + 1d Supertrend uptrend
- Short: price < L3 + volume confirmation + 1d Supertrend downtrend
- Exit: price re-enters H3-L3 range OR 1d Supertrend flips
- Discrete sizing: ±0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average (balanced for trade frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Supertrend for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.full(len(close_1d), np.nan, dtype=float)
    direction = np.full(len(close_1d), 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upperband[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            close_1d[i] = supertrend[i-1]  # for calculation consistency
        
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Align Supertrend direction to 4h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, direction.astype(float))
    
    # Calculate 1d Camarilla levels (H3, L3)
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + rng * 1.1 / 4
    camarilla_l3 = close_1d - rng * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(atr_period, 20)  # Need ATR period and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(supertrend_direction_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 + volume confirmation + 1d Supertrend uptrend
            if (close[i] > h3_aligned[i] and 
                volume_confirm and 
                supertrend_direction_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 + volume confirmation + 1d Supertrend downtrend
            elif (close[i] < l3_aligned[i] and 
                  volume_confirm and 
                  supertrend_direction_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below L3 (mean reversion) OR Supertrend flips down
            if close[i] < l3_aligned[i] or supertrend_direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above H3 (mean reversion) OR Supertrend flips up
            if close[i] > h3_aligned[i] or supertrend_direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dSupertrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0