#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1w HMA(21) is rising AND volume > 1.5x average.
# Short when price breaks below 20-period Donchian low AND 1w HMA(21) is falling AND volume > 1.5x average.
# Exit when price crosses 10-period EMA (trailing exit) or opposite Donchian breakout occurs.
# Uses 1d timeframe for lower trade frequency, Donchian for structure, 1w HMA for trend filter, volume for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1d_Donchian20_1wHMA_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit signal
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 1w data for HMA(21) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on 1w close: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA components
    wma_half = np.array([wma(close_1w[i:i+half_n], half_n) if i+half_n <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    
    # HMA = WMA(2*WMA(half) - WMA(full)), sqrt(n))
    raw_hma = 2 * wma_half - wma_full
    hma_1w = np.array([wma(raw_hma[i:i+sqrt_n], sqrt_n) if i+sqrt_n <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
    
    # Align HTF data to LTF
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema10_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(volume_filter_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate HMA slope for trend direction
        if i >= 1:
            hma_slope = hma_1w_aligned[i] - hma_1w_aligned[i-1]
        else:
            hma_slope = 0
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND HMA rising AND volume confirmation
            if close[i] > donchian_high_aligned[i] and hma_slope > 0 and volume_filter_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND HMA falling AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and hma_slope < 0 and volume_filter_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA10 OR opposite breakout occurs
            if close[i] < ema10_1d_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA10 OR opposite breakout occurs
            if close[i] > ema10_1d_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals