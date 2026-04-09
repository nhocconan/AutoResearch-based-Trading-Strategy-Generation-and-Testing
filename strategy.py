#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and 1d volume spike confirmation
# In strong trends (price > HMA21 on 12h): breakout above/below 20-period Donchian channel with volume confirmation
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: breakout catches strong moves, volume filter avoids false breakouts

name = "4h_12h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21)
    def hull_moving_average(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        
        def wma(data, window):
            if len(data) < window:
                return np.full(len(data), np.nan)
            weights = np.arange(1, window + 1)
            wma_vals = np.convolve(data, weights, mode='valid') / (window * (window + 1) / 2)
            result = np.full(len(data), np.nan)
            result[window-1:] = wma_vals
            return result
        
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt)
        return hma
    
    hma_12h = hull_moving_average(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volume normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
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
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h HMA and 1d volume ratio to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume confirmation: current volume > 2.0 * average 1d volume/ATR ratio
    volume_confirmed = volume > 2.0 * avg_vol_ratio_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(avg_vol_ratio_1d_aligned[i]) or
            np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h HMA
        uptrend = close[i] > hma_12h_aligned[i]
        downtrend = close[i] < hma_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price breaks below lower Donchian or trend changes
            if close[i] < lower_20[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above upper Donchian or trend changes
            if close[i] > upper_20[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above upper Donchian in uptrend with volume confirmation
            if uptrend and close[i] > upper_20[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below lower Donchian in downtrend with volume confirmation
            elif downtrend and close[i] < lower_20[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals