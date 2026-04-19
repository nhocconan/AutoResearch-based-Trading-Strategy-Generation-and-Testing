#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w trend alignment using 1w EMA200 for trend direction,
# 6h Donchian10 breakout for momentum, and volume confirmation. Enters only during 08-20 UTC session.
# Targets 12-30 trades/year (50-120 total over 4 years) with strict entry conditions.
# Uses higher timeframe (1w) trend to avoid whipsaws in bear markets and capture major trends.
name = "6h_1w_EMA200_Donchian10_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA200 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 6h data for Donchian10 breakout (called ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    # Donchian channels: 10-period high/low
    high_10_6h = pd.Series(high_6h).rolling(window=10, min_periods=10).max().values
    low_10_6h = pd.Series(low_6h).rolling(window=10, min_periods=10).min().values
    high_10_6h_aligned = align_htf_to_ltf(prices, df_6h, high_10_6h)
    low_10_6h_aligned = align_htf_to_ltf(prices, df_6h, low_10_6h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(high_10_6h_aligned[i]) or 
            np.isnan(low_10_6h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w EMA200 AND breaks 6h Donchian high with volume
            if (close[i] > ema_200_1w_aligned[i] and 
                close[i] > high_10_6h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA200 AND breaks 6h Donchian low with volume
            elif (close[i] < ema_200_1w_aligned[i] and 
                  close[i] < low_10_6h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1w EMA200 or 6h Donchian low
            if close[i] < ema_200_1w_aligned[i] or close[i] < low_10_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1w EMA200 or 6h Donchian high
            if close[i] > ema_200_1w_aligned[i] or close[i] > high_10_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals