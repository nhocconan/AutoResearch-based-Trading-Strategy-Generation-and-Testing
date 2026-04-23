#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above 20-period Donchian high and 1d ATR(14)/ATR(50) > 1.2 with volume > 1.3x average.
Short when price breaks below 20-period Donchian low and 1d ATR(14)/ATR(50) > 1.2 with volume > 1.3x average.
Exit on opposite Donchian break or ATR ratio < 0.8 (low volatility).
Donchian channels provide clear trend-following structure.
1d ATR ratio filters for expanding volatility regimes to avoid false breakouts in low-vol chop.
Volume confirmation ensures breakout legitimacy.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in expanding volatility regimes.
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
    
    # Load 1d data for ATR volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on 1d data
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_50_1d = calculate_atr(high_1d, low_1d, close_1d, 50)
    
    # Calculate ATR ratio (short-term/long-term volatility)
    atr_ratio_1d = np.where(atr_50_1d != 0, atr_14_1d / atr_50_1d, 1.0)
    
    # Align 1d ATR ratio to 12h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Donchian channels on primary timeframe
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio = atr_ratio_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND ATR ratio > 1.2 (expanding vol) AND volume spike
            if (price > upper and atr_ratio > 1.2 and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND ATR ratio > 1.2 (expanding vol) AND volume spike
            elif (price < lower and atr_ratio > 1.2 and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR ATR ratio < 0.8 (low volatility)
                if (price < lower or atr_ratio < 0.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR ATR ratio < 0.8 (low volatility)
                if (price > upper or atr_ratio < 0.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATRratio_Volume_Breakout"
timeframe = "12h"
leverage = 1.0