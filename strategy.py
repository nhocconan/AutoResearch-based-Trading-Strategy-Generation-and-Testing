#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses the Donchian middle band (20-period mean) or volume drops below average.
Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
Donchian provides objective breakout levels, 1d EMA50 filters for higher timeframe trend,
volume confirmation reduces false breakouts.
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
    
    # Get 6h data for Donchian calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    upper = rolling_max(high_6h, 20)
    lower = rolling_min(low_6h, 20)
    middle = (upper + lower) / 2.0
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_6h, middle)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        middle_val = middle_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > upper band AND close > 1d EMA50 AND volume > 1.5x avg
            if price > upper_val and close[i] > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < lower band AND close < 1d EMA50 AND volume > 1.5x avg
            elif price < lower_val and close[i] < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses middle band OR volume < average
            if price < middle_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses middle band OR volume < average
            if price > middle_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume_1dEMA50_Filter"
timeframe = "6h"
leverage = 1.0