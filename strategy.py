#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1d ATR ratio > 0.8 with volume > 1.3x average.
Short when price breaks below lower Donchian channel and 1d ATR ratio > 0.8 with volume > 1.3x average.
Exit when price crosses the Donchian midpoint or ATR ratio < 0.5 (low volatility).
Donchian channels provide objective trend-following structure from 20-bar high/low.
1d ATR ratio (current ATR / 20-period ATR) filters for sufficient volatility to avoid false breakouts in low-vol periods.
Volume confirmation ensures breakout legitimacy with institutional participation.
Designed for 12h timeframe targeting 12-37 trades per year over 4 years with very low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts when volatility is sufficient.
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
    if len(df_1d) < 20:
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
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.zeros_like(atr_1d)
    for i in range(len(atr_1d)):
        if atr_ma_1d[i] != 0:
            atr_ratio_1d[i] = atr_1d[i] / atr_ma_1d[i]
        else:
            atr_ratio_1d[i] = 0
    
    # Align 1d ATR ratio to 12h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Donchian channels on primary timeframe (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        middle = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2
        
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = donchian_channels(high, low)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(donch_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio = atr_ratio_1d_aligned[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        middle = donch_middle[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND sufficient volatility (ATR ratio > 0.8) AND volume spike
            if (price > upper and atr_ratio > 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND sufficient volatility (ATR ratio > 0.8) AND volume spike
            elif (price < lower and atr_ratio > 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian midpoint OR low volatility (ATR ratio < 0.5)
                if (price < middle or atr_ratio < 0.5):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian midpoint OR low volatility (ATR ratio < 0.5)
                if (price > middle or atr_ratio < 0.5):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATRratio_Volume_Breakout"
timeframe = "12h"
leverage = 1.0