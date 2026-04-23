#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volume spike and choppiness regime filter.
Long when price breaks above Donchian upper band AND 1d ATR(7)/ATR(30) > 1.5 AND CHOP(14) < 38.2 (trending).
Short when price breaks below Donchian lower band AND 1d ATR(7)/ATR(30) > 1.5 AND CHOP(14) < 38.2.
Exit on opposite Donchian band break or CHOP > 61.8 (choppy market).
Donchian bands provide clear breakout levels from recent 20-bar high/low.
1d ATR ratio > 1.5 filters for high volatility breakouts (avoids low-vol false signals).
CHOP < 38.2 ensures we only trade in trending regimes, avoiding whipsaws in chop.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in trending regimes with volatility confirmation.
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
    
    # Load 1d data for ATR ratio and CHOP - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Calculate ATR(7) and ATR(30) for volatility ratio
    atr_7 = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr_30 = calculate_atr(high_1d, low_1d, close_1d, 30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)  # Avoid division by zero
    
    # Align 1d ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Choppiness Index on 1d data
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of TR over period
        tr_sum = np.zeros_like(high)
        for i in range(period, len(high)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(high)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop formula: 100 * log10(sum(tr) / (max_high - min_low)) / log10(period)
        chop = np.zeros_like(high)
        for i in range(period, len(high)):
            if max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when no range
        
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian bands on primary timeframe (12h)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(high)
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
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        chop_val = chop_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper band AND high volatility (ATR ratio > 1.5) AND trending (CHOP < 38.2)
            if (price > upper_val and atr_ratio_val > 1.5 and chop_val < 38.2):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND high volatility AND trending
            elif (price < lower_val and atr_ratio_val > 1.5 and chop_val < 38.2):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower band OR choppy market (CHOP > 61.8)
                if (price < lower_val or chop_val > 61.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper band OR choppy market
                if (price > upper_val or chop_val > 61.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATRratio_CHOP_Volume"
timeframe = "12h"
leverage = 1.0