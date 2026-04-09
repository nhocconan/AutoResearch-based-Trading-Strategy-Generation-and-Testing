#!/usr/bin/env python3
# 12h_donchian_1w_hma_volume_v1
# Hypothesis: 12h Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
# Uses 12h timeframe to target 12-37 trades/year (50-150 over 4 years). Donchian channels provide
# clear breakout signals with defined structure. 1w HMA(21) ensures we only trade in the direction
# of the weekly trend, avoiding counter-trend whipsaws. Volume spike confirms institutional
# participation. Designed to work in both bull and bear markets: breakouts capture momentum
# while the weekly trend filter prevents trading against the dominant trend, reducing false
# signals during ranging periods.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1w_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period using EWMA as approximation
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA for full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Rolling max/min for Donchian channels
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe (completed 12h candle only)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Get 1w HTF data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:  # Need enough for HMA(21)
        return np.zeros(n)
    
    # Calculate 1w HMA(21) for trend
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    
    # Align 1w HMA to 12h timeframe (completed weekly candle only)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Volume spike detection (30-period volume average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian lower band
            if close[i] < lower_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian upper band
            if close[i] > upper_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h Donchian upper band, above 1w HMA, with volume spike
            if (close[i] > upper_12h_aligned[i]) and (close[i] > hma_1w_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h Donchian lower band, below 1w HMA, with volume spike
            elif (close[i] < lower_12h_aligned[i]) and (close[i] < hma_1w_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals