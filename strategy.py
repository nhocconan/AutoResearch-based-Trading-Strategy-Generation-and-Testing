#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h HMA21 rising AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 12h HMA21 falling AND volume > 1.5x 20-period average.
Exit when price crosses Donchian middle band (20-period midpoint).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 25-50 trades/year per symbol.
Donchian channels provide proven structure for breakouts in both bull and bear markets.
12h HMA21 offers smooth trend filter with minimal lag for 4h timeframe alignment.
Volume confirmation ensures only institutional-grade breakouts are taken.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for HMA21 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA21 for trend filter
    hma21_12h = calculate_hma(close_12h, 21)
    
    # Align 12h HMA21 to 4h timeframe
    hma21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma21_12h)
    
    # Donchian(20) on primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    donchian_middle = ((high_roll + low_roll) / 2).values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(hma21_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h HMA21 rising AND volume spike
            if (price > donchian_upper[i] and 
                hma21_12h_aligned[i] > hma21_12h_aligned[i-1] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h HMA21 falling AND volume spike
            elif (price < donchian_lower[i] and 
                  hma21_12h_aligned[i] < hma21_12h_aligned[i-1] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band
            if position == 1 and price < donchian_middle[i]:
                exit_signal = True
            elif position == -1 and price > donchian_middle[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hHMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0