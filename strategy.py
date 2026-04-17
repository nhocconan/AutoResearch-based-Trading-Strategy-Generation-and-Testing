#!/usr/bin/env python3
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
    
    # Calculate ATR(14) for volatility
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                             np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.zeros_like(close)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate daily Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high and low
    high_20 = np.full_like(high_1d, np.nan)
    low_20 = np.full_like(low_1d, np.nan)
    for i in range(20, len(high_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian channels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate daily ATR(14) for volatility filter
    def calculate_atr_series(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                             np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_series = np.zeros_like(close)
        atr_series[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_series[i] = (atr_series[i-1] * (period-1) + tr[i]) / period
        return atr_series
    
    atr_1d = calculate_atr_series(high_1d, low_1d, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 20)  # Donchian channels and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 0.5 * daily ATR (avoid extremely low vol)
        vol_filter = atr[i] > (0.5 * atr_1d_aligned[i])
        
        # Volume filter: current volume > 1.5 * 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20_aligned[i]
        breakout_down = close[i] < low_20_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume filter + volatility filter
            if breakout_up and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume filter + volatility filter
            elif breakout_down and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below the lower Donchian band
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above the upper Donchian band
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_Volume_VolatilityFilter"
timeframe = "6h"
leverage = 1.0