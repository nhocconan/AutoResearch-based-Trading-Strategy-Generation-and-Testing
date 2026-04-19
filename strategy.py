#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d trend (EMA50) and 1w momentum (Donchian55) for trend alignment.
# Enters on breakouts of 4h Donchian20 with volume confirmation only when aligned with higher timeframe trends.
# Uses 08-20 UTC session filter to avoid low-volume periods. Designed to work in both bull and bear markets
# by following higher timeframe trends and avoiding counter-trend trades. Targets 20-40 trades/year.
name = "4h_1d1w_EMA50_Donchian55_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for Donchian55 momentum (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_55_1w = pd.Series(high_1w).rolling(window=55, min_periods=55).max().values
    low_55_1w = pd.Series(low_1w).rolling(window=55, min_periods=55).min().values
    high_55_1w_aligned = align_htf_to_ltf(prices, df_1w, high_55_1w)
    low_55_1w_aligned = align_htf_to_ltf(prices, df_1w, low_55_1w)
    
    # Get 4h data for Donchian20 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Volume filter: volume > 1.5 * 20-period average (using 4h volume)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_55_1w_aligned[i]) or 
            np.isnan(low_55_1w_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA50 AND above 1w Donchian55 low (bullish alignment) 
            # AND breaks 4h Donchian20 high with volume
            if (close[i] > ema_50_1d_aligned[i] and 
                close[i] > low_55_1w_aligned[i] and 
                close[i] > high_20_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 AND below 1w Donchian55 high (bearish alignment)
            # AND breaks 4h Donchian20 low with volume
            elif (close[i] < ema_50_1d_aligned[i] and 
                  close[i] < high_55_1w_aligned[i] and 
                  close[i] < low_20_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 4h Donchian20 low or 1d EMA50
            if close[i] < low_20_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 4h Donchian20 high or 1d EMA50
            if close[i] > high_20_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals