#!/usr/bin/env python3
# 6h_1dATR_Based_Volume_Breakout_Trend_Filter
# Uses daily ATR for volatility-adjusted breakout levels and volume confirmation
# Designed for 6h timeframe to capture volatility breakouts aligned with daily trend
# Works in both bull and bear markets by following daily trend and filtering with volume spikes

name = "6h_1dATR_Based_Volume_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for ATR, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # ATR (14-period) using Wilder's smoothing
    atr_14 = np.zeros(len(high_1d))
    atr_14[13] = np.mean(tr[1:14])  # Initialize with simple average
    for i in range(14, len(high_1d)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily ATR-based breakout levels (similar to Donchian but ATR-adjusted)
    # Upper band: highest high of last 20 days + 0.5 * ATR
    # Lower band: lowest low of last 20 days - 0.5 * ATR
    lookback = 20
    upper_band = np.zeros(len(high_1d))
    lower_band = np.zeros(len(high_1d))
    
    for i in range(lookback-1, len(high_1d)):
        highest_high = np.max(high_1d[i-lookback+1:i+1])
        lowest_low = np.min(low_1d[i-lookback+1:i+1])
        upper_band[i] = highest_high + 0.5 * atr_14[i]
        lower_band[i] = lowest_low - 0.5 * atr_14[i]
    
    # Align ATR-based bands to 6h timeframe
    upper_band_6h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_6h = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume filter (20-period MA) with threshold for significant volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_band_6h[i]) or np.isnan(lower_band_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: break above upper band with uptrend and volume confirmation
            if close[i] > upper_band_6h[i] and close[i] > ema_34_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below lower band with downtrend and volume confirmation
            elif close[i] < lower_band_6h[i] and close[i] < ema_34_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA34 or breaks below lower band
            if bars_since_entry >= 2 and (close[i] < ema_34_6h[i] or close[i] < lower_band_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA34 or breaks above upper band
            if bars_since_entry >= 2 and (close[i] > ema_34_6h[i] or close[i] > upper_band_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals