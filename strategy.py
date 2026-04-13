#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d ATR-based volatility filter and Donchian channel breakout.
# Long: Price breaks above Donchian(20) upper band + 1d ATR(10) > 1d ATR(30) (volatility expansion).
# Short: Price breaks below Donchian(20) lower band + 1d ATR(10) > 1d ATR(30).
# Uses volatility expansion to confirm breakout strength, reducing false signals.
# Works in bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10) and ATR(30) for 1d
    atr10_1d = np.full(len(close_1d), np.nan)
    atr30_1d = np.full(len(close_1d), np.nan)
    
    # Wilder's smoothing for ATR
    for i in range(len(close_1d)):
        if i < 10:
            atr10_1d[i] = np.nan
        elif i == 10:
            atr10_1d[i] = np.nanmean(tr[1:11])
        else:
            atr10_1d[i] = (atr10_1d[i-1] * 9 + tr[i]) / 10
            
        if i < 30:
            atr30_1d[i] = np.nan
        elif i == 30:
            atr30_1d[i] = np.nanmean(tr[1:31])
        else:
            atr30_1d[i] = (atr30_1d[i-1] * 29 + tr[i]) / 30
    
    # Volatility expansion: ATR(10) > ATR(30)
    vol_expansion = atr10_1d > atr30_1d
    
    # Donchian channel (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Align 1d volatility filter to 4h
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_exp = vol_expansion_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: price breaks above upper band + volatility expansion
            if (price > upper and vol_exp):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band + volatility expansion
            elif (price < lower and vol_exp):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band (opposite side)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band (opposite side)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ATR_Volatility_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0