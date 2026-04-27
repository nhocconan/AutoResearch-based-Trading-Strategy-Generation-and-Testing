#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with daily volume confirmation and 1-day ATR volatility filter.
Enters long when price breaks above 20-period high with volume > 1.5x daily average and ATR(14) > 0.5 * ATR(50).
Enters short when price breaks below 20-period low with volume > 1.5x daily average and ATR(14) > 0.5 * ATR(50).
Uses ATR-based stoploss to limit drawdown. Designed to capture trending moves while avoiding choppy markets.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, length):
    """Average True Range"""
    if length <= 0:
        return np.full_like(high, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    atr[length-1] = np.mean(tr[:length])
    for i in range(length, len(tr)):
        atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) and ATR(50) on 1d
    atr_14_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_50_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), volume MA (20), ATR(50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_14 = atr_14_1d_aligned[i]
        atr_50 = atr_50_1d_aligned[i]
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50) (ensures sufficient volatility)
        vol_filter = atr_14 > 0.5 * atr_50
        
        # Volume filter: volume > 1.5x daily average
        volume_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above 20-period high with volume + volatility
            if price_now > high_20[i] and volume_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below 20-period low with volume + volatility
            elif price_now < low_20[i] and volume_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 20-period low or volatility drops
            if price_now < low_20[i] or atr_14 < 0.3 * atr_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 20-period high or volatility drops
            if price_now > high_20[i] or atr_14 < 0.3 * atr_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_1dVolume_ATRFilter"
timeframe = "4h"
leverage = 1.0