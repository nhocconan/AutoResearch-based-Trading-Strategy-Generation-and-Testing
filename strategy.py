#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA200 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA200 AND volume > 1.8x 20-period average.
Exit when price crosses the Donchian middle band (20-period mean).
Uses 1d HTF for trend filter to improve robustness in both bull and bear markets.
Target: 12-37 trades/year per symbol to avoid fee drag.
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
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    def rolling_mean(arr, window):
        """Rolling mean"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    upper_band = rolling_max(high_12h, 20)
    lower_band = rolling_min(low_12h, 20)
    middle_band = rolling_mean(close_12h, 20)
    
    # Get 1d data for EMA200 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_band)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        ema200 = ema200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + price > 1d EMA200 + volume spike
            if price > upper and price > ema200 and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Breakout below lower band + price < 1d EMA200 + volume spike
            elif price < lower and price < ema200 and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit condition for long: Price crosses middle band (mean reversion)
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit condition for short: Price crosses middle band (mean reversion)
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0