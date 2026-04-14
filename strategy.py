#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter and volatility context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly ATR for volatility context
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr1w[0] = high_1w[0] - low_1w[0]  # first period
    atr1w = pd.Series(tr1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr1w_aligned = align_htf_to_ltf(prices, df_1w, atr1w)
    
    # Load daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channel (20-day) on daily data
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (will be used for 1d entries)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly volatility filter: avoid extremely low volatility periods
        if atr1w_aligned[i] < np.nanpercentile(atr1w_aligned[:i+1], 20):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly EMA50 AND breaks above 20-day high
            if (close[i] > ema50_1w_aligned[i] and 
                close[i] > highest_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly EMA50 AND breaks below 20-day low
            elif (close[i] < ema50_1w_aligned[i] and 
                  close[i] < lowest_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 OR 20-day low
            if (close[i] < ema50_1w_aligned[i] or
                close[i] < lowest_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 OR 20-day high
            if (close[i] > ema50_1w_aligned[i] or
                close[i] > highest_20_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyEMA50_Donchian20_Breakout_v1"
timeframe = "1d"
leverage = 1.0