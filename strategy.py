#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1d ATR ratio > 0.8 with volume > 1.3x average.
Short when price breaks below lower Donchian channel and 1d ATR ratio > 0.8 with volume > 1.3x average.
Exit on opposite Donchian break or ATR ratio < 0.5 (low volatility regime).
Donchian channels provide objective trend-following structure proven on SOLUSDT.
1d ATR ratio (current ATR / 20-period ATR mean) filters for sufficient volatility to avoid false breakouts in low-vol regimes.
Volume confirmation ensures breakout legitimacy.
Designed for 4h timeframe targeting 75-200 total trades over 4 years with controlled frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in sufficient volatility regimes.
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
    
    # Load 1d data for ATR regime filter - ONCE before loop
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
    
    # Calculate 20-period ATR mean for regime filter
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.zeros_like(atr_1d)
    for i in range(len(atr_1d)):
        if atr_ma_20[i] > 0:
            atr_ratio_1d[i] = atr_1d[i] / atr_ma_20[i]
        else:
            atr_ratio_1d[i] = 0
    
    # Align 1d ATR ratio to 4h timeframe
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
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_1d_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND ATR ratio > 0.8 (sufficient vol) AND volume spike
            if (price > upper_val and atr_ratio_val > 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND ATR ratio > 0.8 (sufficient vol) AND volume spike
            elif (price < lower_val and atr_ratio_val > 0.8 and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR ATR ratio < 0.5 (low volatility)
                if (price < lower_val or atr_ratio_val < 0.5):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR ATR ratio < 0.5 (low volatility)
                if (price > upper_val or atr_ratio_val < 0.5):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Ratio_Volume"
timeframe = "4h"
leverage = 1.0