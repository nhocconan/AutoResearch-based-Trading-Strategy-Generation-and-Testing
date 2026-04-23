#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band and close > 12h HMA21 (uptrend) with volume > 1.8x average.
Short when price breaks below 20-period Donchian lower band and close < 12h HMA21 (downtrend) with volume > 1.8x average.
Exit on opposite Donchian band break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear breakout levels, 12h HMA filters medium-term trend without whipsaw, volume confirms strength.
Designed to work in both bull and bear markets by following the 12h trend direction while avoiding false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels from previous 4h bar (avoid look-ahead)
    # Upper band = 20-period high, Lower band = 20-period low
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for HMA21 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).rolling(window=half_n, min_periods=half_n).apply(
        lambda x: np.average(x, weights=np.arange(1, half_n + 1)), raw=True
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, 22)), raw=True
    ).values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.average(x, weights=np.arange(1, sqrt_n + 1)), raw=True
    ).values
    
    # Align HTF indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        hma_val = hma_21_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 12h HMA21 (uptrend) AND volume spike
            if (price > upper_val and price > hma_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND price < 12h HMA21 (downtrend) AND volume spike
            elif (price < lower_val and price < hma_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower OR trend reversal
                if (price < lower_val or price < hma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper OR trend reversal
                if (price > upper_val or price > hma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hHMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0