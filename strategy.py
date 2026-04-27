#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above S3 with 12h EMA50 uptrend and volume > 1.8x average.
# Short when price breaks below R3 with 12h EMA50 downtrend and volume > 1.8x average.
# Exit when price crosses the pivot point (PP).
# Target: 20-40 trades/year to avoid fee drag. Camarilla levels provide strong support/resistance.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h EMA50 and volume MA20
    start_idx = max(ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            i < 1):  # need previous day for Camarilla
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous day
        prev_high = high_1d[i - 1] if i - 1 < len(high_1d) else high_1d[-1]
        prev_low = low_1d[i - 1] if i - 1 < len(low_1d) else low_1d[-1]
        prev_close = close_1d[i - 1] if i - 1 < len(close_1d) else close_1d[-1]
        
        # Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        pp = (prev_high + prev_low + prev_close) / 3
        s3 = prev_close - (range_val * 1.1 / 2)
        r3 = prev_close + (range_val * 1.1 / 2)
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: break above S3 with 12h EMA50 uptrend and volume
            if (price > s3 and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below R3 with 12h EMA50 downtrend and volume
            elif (price < r3 and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_S3R3_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0