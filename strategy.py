#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d trend alignment using 4h EMA50 and 1d EMA200 for trend direction,
# 1h Donchian10 breakout for momentum, and volume confirmation. Enters only during 08-20 UTC session.
# Uses 4h trend as primary filter, 1d trend as secondary filter to avoid counter-trend trades.
# Targets 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends.
name = "1h_4h_1d_EMA50_EMA200_Donchian10_Volume"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for EMA200 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1h data for Donchian10 breakout (called ONCE before loop)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above both 4h EMA50 and 1d EMA200 AND breaks 1h Donchian high with volume
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                close[i] > high_10[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below both 4h EMA50 and 1d EMA200 AND breaks 1h Donchian low with volume
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  close[i] < low_10[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below either 4h EMA50 or 1d EMA200 or 1h Donchian low
            if close[i] < ema_50_4h_aligned[i] or close[i] < ema_200_1d_aligned[i] or close[i] < low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above either 4h EMA50 or 1d EMA200 or 1h Donchian high
            if close[i] > ema_50_4h_aligned[i] or close[i] > ema_200_1d_aligned[i] or close[i] > high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals