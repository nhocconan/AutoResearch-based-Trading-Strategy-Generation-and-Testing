#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d Volume Spike + Volatility Filter
Long: Price > Donchian High(20) + 1d volume > 1.5x 20-period avg + ATR(14) < 0.03 * price
Short: Price < Donchian Low(20) + 1d volume > 1.5x 20-period avg + ATR(14) < 0.03 * price
Exit: Opposite Donchian break or ATR(14) > 0.05 * price (high volatility)
Designed to capture strong breakouts with volume confirmation in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20)
    vol_sma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_val = vol_1d[i]
        vol_sma_val = vol_sma_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Price > Donchian High + 1d volume spike + low volatility
            if price > donchian_high[i] and vol_1d_val > 1.5 * vol_sma_val and atr_val < 0.03 * price:
                signals[i] = 0.25
                position = 1
            # Short: Price < Donchian Low + 1d volume spike + low volatility
            elif price < donchian_low[i] and vol_1d_val > 1.5 * vol_sma_val and atr_val < 0.03 * price:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price < Donchian Low or high volatility
            if price < donchian_low[i] or atr_val > 0.05 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price > Donchian High or high volatility
            if price > donchian_high[i] or atr_val > 0.05 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dVol_Spike_VolatilityFilter"
timeframe = "12h"
leverage = 1.0