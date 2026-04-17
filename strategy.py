#!/usr/bin/env python3
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
    
    # Get 1d data for HTF
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA200 for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d ATR for volatility measurement
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate 12h Bollinger Bands for squeeze detection
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Bollinger Bands width (normalized by ATR)
    sma20_12h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20_12h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width_12h = (4 * std20_12h) / sma20_12h  # Width as percentage of middle band
    bb_width_ma50 = pd.Series(bb_width_12h).rolling(window=50, min_periods=50).mean().values
    
    # Bollinger Squeeze: BB width below 50-day average
    bb_squeeze = bb_width_12h < bb_width_ma50
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Bollinger squeeze condition (must be in squeeze to enter)
        if not bb_squeeze[i]:
            # Exit if not in squeeze
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price touches lower BB and above EMA200
        long_condition = (low[i] <= lower_bb_1d_aligned[i]) and (close[i] > ema200_1d_aligned[i])
        
        # Short conditions: price touches upper BB and below EMA200
        short_condition = (high[i] >= upper_bb_1d_aligned[i]) and (close[i] < ema200_1d_aligned[i])
        
        if position == 0:
            # Enter long
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Enter short
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: price crosses above EMA200 or touches upper BB
            if close[i] >= ema200_1d_aligned[i] or high[i] >= upper_bb_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below EMA200 or touches lower BB
            if close[i] <= ema200_1d_aligned[i] or low[i] <= lower_bb_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_EMA200_Reversion"
timeframe = "12h"
leverage = 1.0