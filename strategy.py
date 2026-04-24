#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for ATR-based volatility filter.
- Donchian channels calculated from prior 20-period 12h high/low.
- Breakout logic: long when price closes above upper Donchian with volume spike and ATR > 1d ATR MA,
                  short when price closes below lower Donchian with volume spike and ATR > 1d ATR MA.
- ATR filter: only trade when current 12h ATR(14) > 1.0 * 1d ATR(14) MA (ensures sufficient volatility).
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA.
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: volatility filter avoids low-volatility chop, Donchian breakouts capture momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # first bar
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h ATR(14) for current volatility
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = high[0] - low[0]
    tr2_12h[0] = np.abs(high[0] - close[0])
    tr3_12h[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels from prior 20-period 12h high/low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (shifted by 1 to use prior bar's values)
    donchian_upper_aligned = np.roll(donchian_upper, 1)
    donchian_lower_aligned = np.roll(donchian_lower, 1)
    donchian_upper_aligned[0] = np.nan
    donchian_lower_aligned[0] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Volatility filter: 12h ATR > 1.0 * 1d ATR MA
    vol_filter = atr_14_12h > (1.0 * atr_14_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14)  # Need Donchian(20) and ATR(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian AND volume spike AND vol filter
            if close[i] > donchian_upper_aligned[i] and volume_spike[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower Donchian AND volume spike AND vol filter
            elif close[i] < donchian_lower_aligned[i] and volume_spike[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to lower Donchian or reverse signal
            if close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to upper Donchian or reverse signal
            if close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0