#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d volatility regime filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND 1d ATR ratio (ATR7/ATR30) < 0.8 (low vol regime) AND volume > 1.3x average.
Short when price breaks below 20-period Donchian low AND 1d ATR ratio < 0.8 AND volume > 1.3x average.
Exit on opposite Donchian break or ATR ratio > 1.2 (volatility expansion).
Low volatility regimes precede breakouts; volume confirms legitimacy. Works in both bull/bear by capturing expansion after contraction.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with strict entry conditions to minimize fee drag.
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
    
    # Load 1d data for volatility regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d data
    def calculate_atr(high, low, close, period):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr7_1d = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr30_1d = calculate_atr(high_1d, low_1d, close_1d, 30)
    
    # ATR ratio: ATR7 / ATR30 (low when < 0.8 indicates low volatility regime)
    atr_ratio_1d = np.zeros_like(atr7_1d)
    for i in range(len(atr_ratio_1d)):
        if atr30_1d[i] != 0:
            atr_ratio_1d[i] = atr7_1d[i] / atr30_1d[i]
        else:
            atr_ratio_1d[i] = 0
    
    # Align 1d ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    def calculate_donchian(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, 20)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio = atr_ratio_1d_aligned[i]
        upper = donchian_20_upper[i]
        lower = donchian_20_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper DONCHIAN AND low vol regime (ATR ratio < 0.8) AND volume spike
            if (price > upper and atr_ratio < 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower DONCHIAN AND low vol regime AND volume spike
            elif (price < lower and atr_ratio < 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower DONCHIAN OR ATR ratio > 1.2 (vol expansion)
                if (price < lower or atr_ratio > 1.2):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper DONCHIAN OR ATR ratio > 1.2
                if (price > upper or atr_ratio > 1.2):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dATRratio_Volume_Breakout"
timeframe = "6h"
leverage = 1.0