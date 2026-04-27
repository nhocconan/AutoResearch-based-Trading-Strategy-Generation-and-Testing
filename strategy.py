#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and 1-day ATR filter.
Trades breakouts in direction of daily trend (close > SMA50) when volume > 1.5x 4h average.
ATR-based stoploss to manage risk. Designed to work in both bull and bear markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day SMA(50) for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Get 4-hour data for ATR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour ATR(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Get 4-hour data for volume filter
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, ATR, and daily SMA
    start_idx = max(20, 20, 14, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(sma_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        atr = atr_14_aligned[i]
        trend_1d = sma_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above upper Donchian + volume + daily uptrend
            if close[i] > high_20_val and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below lower Donchian + volume + daily downtrend
            elif close[i] < low_20_val and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below lower Donchian or ATR-based stop
            if close[i] < low_20_val or close[i] < (signals[i-1] * 0 + entry_price - 2 * atr):  # Simplified stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above upper Donchian or ATR-based stop
            if close[i] > high_20_val or close[i] > (signals[i-1] * 0 + entry_price + 2 * atr):  # Simplified stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Volume_1dTrendFilter_ATRStop"
timeframe = "4h"
leverage = 1.0