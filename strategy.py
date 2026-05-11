#!/usr/bin/env python3
name = "6h_1w_1d_WickReversal"
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
    
    # Get 1d and 1w data (HTF)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 4:
        return np.zeros(n)
    
    # 1d High/Low for Wick Reversal detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d upper and lower wick percentages
    body_size = np.abs(close_1d - np.roll(close_1d, 1))
    upper_wick = high_1d - np.maximum(close_1d, np.roll(close_1d, 1))
    lower_wick = np.minimum(close_1d, np.roll(close_1d, 1)) - low_1d
    
    # Avoid division by zero
    body_size_safe = np.where(body_size == 0, 1, body_size)
    upper_wick_pct = upper_wick / body_size_safe
    lower_wick_pct = lower_wick / body_size_safe
    
    # Wick reversal signals: long when lower wick > 2x body, short when upper wick > 2x body
    bullish_wick = lower_wick_pct > 2.0
    bearish_wick = upper_wick_pct > 2.0
    
    # Shift to get previous day's signal (avoid look-ahead)
    bullish_wick_prev = np.roll(bullish_wick, 1)
    bearish_wick_prev = np.roll(bearish_wick, 1)
    bullish_wick_prev[0] = False
    bearish_wick_prev[0] = False
    
    # Align wick reversal signals to 6h timeframe
    bullish_wick_aligned = align_htf_to_ltf(prices, df_1d, bullish_wick_prev.astype(float))
    bearish_wick_aligned = align_htf_to_ltf(prices, df_1d, bearish_wick_prev.astype(float))
    
    # 1w trend filter: price above/below 20-period EMA
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    volume_filter = vol_ratio > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_wick_aligned[i]) or np.isnan(bearish_wick_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_filter[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish wick reversal on 1d + price above 1w EMA20 + volume filter
            if (bullish_wick_aligned[i] > 0.5 and 
                close[i] > ema_20_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish wick reversal on 1d + price below 1w EMA20 + volume filter
            elif (bearish_wick_aligned[i] > 0.5 and 
                  close[i] < ema_20_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite wick signal or trend change
            if position == 1:
                # Exit long: bearish wick reversal or price below 1w EMA20
                if (bearish_wick_aligned[i] > 0.5) or (close[i] < ema_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish wick reversal or price above 1w EMA20
                if (bullish_wick_aligned[i] > 0.5) or (close[i] > ema_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals