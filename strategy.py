#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volatility filter and volume spike confirmation.
- Uses 12h Camarilla pivot levels (H3/L3) for institutional breakout signals
- 1d ATR(14) as volatility filter (breakouts only when volatility is elevated)
- Volume > 1.8x 20-period average for confirmation
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via volatility filter and volume confirmation
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h Camarilla pivot levels (H3, L3)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    close_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    pivot = (highest_20 + lowest_20 + close_20) / 3.0
    range_20 = highest_20 - lowest_20
    H3 = pivot + range_20 * 1.1 / 2.0
    L3 = pivot - range_20 * 1.1 / 2.0
    
    # 1d data for ATR(14) volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Camarilla, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(H3[i]) or
            np.isnan(L3[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Volatility filter: ATR above its 20-period average (elevated volatility)
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr_14_1d_aligned[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Camarilla H3/L3 breakout signals
        breakout_up = close[i] > H3[i-1]  # Close above prior H3 level
        breakout_down = close[i] < L3[i-1]  # Close below prior L3 level
        
        if position == 0:
            # Long: 12h Camarilla H3 breakout up AND volume confirmation AND volatility filter
            if breakout_up and volume_confirm and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: 12h Camarilla L3 breakout down AND volume confirmation AND volatility filter
            elif breakout_down and volume_confirm and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 12h Camarilla L3 breakdown OR volatility drops
            if breakout_down or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 12h Camarilla H3 breakout OR volatility drops
            if breakout_up or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0