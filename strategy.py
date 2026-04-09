#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation
# and ATR-based position sizing. In bull markets, breakouts capture strong trends.
# In bear markets, false breakouts fade quickly, limiting losses. Weekly timeframe
# reduces noise and whipsaws. Volume confirmation ensures breakouts have conviction.
# Discrete position sizing (0.25) limits trades to ~10-25/year to minimize fee drag.

name = "1d_1w_donchian_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_20 = rolling_max(high_1w, 20)
    lower_20 = rolling_min(low_1w, 20)
    
    # Calculate 1w ATR(14) for volatility normalization
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Calculate 1w volume moving average (20-period)
    def sma(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    vol_ma_20 = sma(volume_1w, 20)
    
    # Align 1w indicators to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price closes below midpoint of Donchian channel
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price closes above midpoint of Donchian channel
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above upper Donchian with volume confirmation
            if high[i] > upper_20_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short on breakdown below lower Donchian with volume confirmation
            elif low[i] < lower_20_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals