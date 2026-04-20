#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Trend_Filter_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w: Trend filter (HMA21) ===
    close_1w = df_1w['close'].values
    hma_21_1w = calculate_hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        hma_val = hma_21_1w_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(hma_val) or np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + uptrend + volume
            if (high_val > donchian_high_val and
                close_val > hma_val and
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + downtrend + volume
            elif (low_val < donchian_low_val and
                  close_val < hma_val and
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend change
            if (low_val < donchian_low_val or
                close_val < hma_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend change
            if (high_val > donchian_high_val or
                close_val > hma_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(arr, period):
    """Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma2 = wma(arr, half_period)
    wma1 = wma(arr, period)
    raw_hma = 2 * wma2 - wma1
    hma = wma(raw_hma, sqrt_period)
    return hma

def wma(arr, period):
    """Weighted Moving Average"""
    if period <= 0:
        return np.full_like(arr, np.nan)
    weights = np.arange(1, period + 1)
    wma_vals = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        wma_vals[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
    return wma_vals