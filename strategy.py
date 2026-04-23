#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR ratio filter and volume confirmation.
Long when price breaks above Donchian upper band and 1d ATR(7)/ATR(30) > 1.2 with volume > 1.5x average.
Short when price breaks below Donchian lower band and 1d ATR(7)/ATR(30) > 1.2 with volume > 1.5x average.
Exit on opposite Donchian break or ATR ratio < 0.8 (volatility contraction).
ATR ratio filter identifies expanding volatility environments conducive to breakouts.
Designed for 4h timeframe targeting 75-200 total trades over 4 years with controlled frequency to minimize fee drag.
Works in both bull and bear markets by trading breakouts in expanding volatility regimes.
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
    
    # Load 1d data for ATR ratio filter - ONCE before loop
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
    
    atr_7 = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr_30 = calculate_atr(high_1d, low_1d, close_1d, 30)
    
    # Avoid division by zero
    atr_ratio = np.zeros_like(atr_7)
    mask = atr_30 != 0
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    def donchian_channel(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channel(high, low, 20)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        upper_val = donch_upper[i]
        lower_val = donch_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper band AND ATR ratio > 1.2 (expanding vol) AND volume spike
            if (price > upper_val and atr_ratio_val > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND ATR ratio > 1.2 (expanding vol) AND volume spike
            elif (price < lower_val and atr_ratio_val > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower band OR ATR ratio < 0.8 (vol contraction)
                if (price < lower_val or atr_ratio_val < 0.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper band OR ATR ratio < 0.8 (vol contraction)
                if (price > upper_val or atr_ratio_val < 0.8):
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