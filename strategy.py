#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ChandelierExit_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 days for calculations
        return np.zeros(n)
    
    # === 1d: Calculate ATR for Chandelier Exit ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(22) - approximately 1 month
    atr = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    
    # Chandelier Exit: 3 * ATR below highest high (long) / above lowest low (short)
    highest_high = pd.Series(high_1d).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low_1d).rolling(window=22, min_periods=22).min().values
    
    chandelier_long = highest_high - 3.0 * atr  # Exit long when price falls below
    chandelier_short = lowest_low + 3.0 * atr   # Exit short when price rises above
    
    # Align 1d indicators to 12h timeframe
    chandelier_long_aligned = align_htf_to_ltf(prices, df_1d, chandelier_long)
    chandelier_short_aligned = align_htf_to_ltf(prices, df_1d, chandelier_short)
    
    # === 12h: Price action and volume confirmation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = close[i]
        long_exit = chandelier_long_aligned[i]
        short_exit = chandelier_short_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(long_exit) or np.isnan(short_exit) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Chandelier long exit with volume confirmation
            if (close_val > long_exit and   # Above long exit level
                vol_ratio_val > 1.5):       # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below Chandelier short exit with volume confirmation
            elif (close_val < short_exit and   # Below short exit level
                  vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below Chandelier long exit
            if close_val < long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above Chandelier short exit
            if close_val > short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals