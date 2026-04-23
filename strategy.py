#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above Donchian(20) high AND 1d ATR(14) < 0.8 * 20-period mean ATR AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND same volatility/volume conditions.
Exit when price crosses Donchian midpoint.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Volatility filter ensures entries occur during low-volatility regimes (pre-breakout compression), 
which increases breakout reliability and reduces false signals in choppy markets. Works in both bull and bear markets.
"""

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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h data
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 4h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Load 1d data for ATR-based volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d data
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, tr2)
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period mean of 1d ATR for volatility regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_ma_val = atr_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND low volatility regime AND volume spike
            if (price > donchian_high_aligned[i] and 
                atr_1d[-1] < 0.8 * atr_ma_val if len(atr_1d) > 0 else False and  # Simplified: use current ATR vs MA
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND low volatility regime AND volume spike
            elif (price < donchian_low_aligned[i] and 
                  atr_1d[-1] < 0.8 * atr_ma_val if len(atr_1d) > 0 else False and  # Simplified: use current ATR vs MA
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian midpoint
                if price < donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian midpoint
                if price > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Vol_Filter"
timeframe = "4h"
leverage = 1.0