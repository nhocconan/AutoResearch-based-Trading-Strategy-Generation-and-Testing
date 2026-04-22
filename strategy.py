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
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above weekly EMA200 (long-term uptrend)
            if (close[i] > high_20_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume AND below weekly EMA200 (long-term downtrend)
            elif (close[i] < low_20_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite Donchian level
            if position == 1:
                # Exit long: Price closes below Donchian low
                if close[i] < low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above Donchian high
                if close[i] > high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Donchian20_WeeklyEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0