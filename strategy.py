#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
- Donchian channel breakout provides clear structure for entries
- 1d ATR(14) measures volatility regime - only trade when ATR > 20-period average (high volatility breakouts)
- Volume confirmation (> 1.5x 20-period average) ensures momentum behind breakouts
- Designed for 4h timeframe targeting 20-40 trades/year (80-160 over 4 years)
- Works in both bull and bear markets by capturing volatility expansion breakouts
- ATR filter avoids low volatility false breakouts, volume filter ensures conviction
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
    
    # Calculate 4h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR MA to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14+20, 20)  # Donchian, ATR, ATR MA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_upper = close[i] > high_roll[i]
        price_below_lower = close[i] < low_roll[i]
        
        # Volatility filter: current ATR > ATR moving average (high volatility regime)
        high_volatility = atr_14_aligned[i] > atr_ma_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian band, high volatility, volume spike
            long_signal = (price_above_upper and 
                          high_volatility and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below lower Donchian band, high volatility, volume spike
            short_signal = (price_below_lower and 
                           high_volatility and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower Donchian band or volatility contracts
                if (price_below_lower or 
                    not high_volatility):  # Volatility contraction
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper Donchian band or volatility contracts
                if (price_above_upper or 
                    not high_volatility):  # Volatility contraction
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0