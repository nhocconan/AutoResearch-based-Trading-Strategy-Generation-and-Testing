#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Fibonacci_Extension_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Fibonacci swing points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA200 for higher timeframe trend
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily swing high/low for Fibonacci extension
    # Look back 50 days for significant swing
    lookback = 50
    highest_high = np.full(len(df_1d), np.nan)
    lowest_low = np.full(len(df_1d), np.nan)
    
    for i in range(lookback, len(df_1d)):
        highest_high[i] = np.max(df_1d['high'].iloc[i-lookback:i])
        lowest_low[i] = np.min(df_1d['low'].iloc[i-lookback:i])
    
    # Calculate Fibonacci extension levels (127.2% and 161.8%)
    range_ = highest_high - lowest_low
    fib_ext_127 = lowest_low + 1.272 * range_
    fib_ext_162 = lowest_low + 1.618 * range_
    
    # Align to 6h timeframe
    fib_ext_127_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_127)
    fib_ext_162_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_162)
    
    # Volume confirmation - 24-period average volume (4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 300
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(fib_ext_127_aligned[i]) or np.isnan(fib_ext_162_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 127.2% Fib extension + daily trend up + weekly trend up + volume
            if (close[i] > fib_ext_127_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                ema_50_1d_aligned[i] > ema_200_1w_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: break below 127.2% Fib extension + daily trend down + weekly trend down + volume
            elif (close[i] < fib_ext_127_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  ema_50_1d_aligned[i] < ema_200_1w_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions: break below 161.8% extension OR trend reversal
            if close[i] < fib_ext_162_aligned[i] or ema_50_1d_aligned[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: break above 161.8% extension OR trend reversal
            if close[i] > fib_ext_162_aligned[i] or ema_50_1d_aligned[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals