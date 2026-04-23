#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
- Donchian channel (20-period high/low) from 12h timeframe for breakout signals
- 1d ATR(14) normalized by price defines volatility regime: only trade when ATR/price > 0.02 (sufficient volatility)
- Volume confirmation (> 1.5x 20-period average) ensures breakout has momentum
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading breakouts with volatility filter
- ATR regime filter avoids choppy markets where breakouts fail
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
    
    # Calculate 12h Donchian channel (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma[i]) or
            close[i] == 0):  # Avoid division by zero
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR/price > 0.02 (sufficient volatility for breakouts)
        vol_regime = (atr_14_aligned[i] / close[i]) > 0.02
        
        # Determine breakout conditions
        price_above_donchian_high = close[i] > high_ma[i]
        price_below_donchian_low = close[i] < low_ma[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, vol regime, volume spike
            long_signal = (price_above_donchian_high and 
                          vol_regime and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low, vol regime, volume spike
            short_signal = (price_below_donchian_low and 
                           vol_regime and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility regime fails
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or vol regime fails
                if (price_below_donchian_low or 
                    not vol_regime):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or vol regime fails
                if (price_above_donchian_high or 
                    not vol_regime):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_VolumeRegime"
timeframe = "12h"
leverage = 1.0