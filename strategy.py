#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Breakout of 4h Donchian(20) channel with volume confirmation and 1d trend filter.
Long when price breaks above upper band + volume > 1.5x avg + 1d close > 1d open (bullish day).
Short when price breaks below lower band + volume > 1.5x avg + 1d close < 1d open (bearish day).
Exit when price returns to 4h 20-period EMA. Designed for low trade frequency (~20-50/year)
to minimize fee drag. Works in bull/bear markets via 1d trend filter aligning with breakout direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_open = df_daily['open'].values
    daily_bullish = daily_close > daily_open  # bullish day
    daily_bearish = daily_close < daily_open  # bearish day
    
    # Align daily trend to 4h
    daily_bullish_aligned = align_htf_to_ltf(prices, df_daily, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_daily, daily_bearish.astype(float))
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
        else:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * vol_ma)
    
    # 4h 20-period EMA for exit
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: breakout above upper band + volume + bullish day
            if price > upper[i] and vol_ok and daily_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + volume + bearish day
            elif price < lower[i] and vol_ok and daily_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to EMA20
            if price < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to EMA20
            if price > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0