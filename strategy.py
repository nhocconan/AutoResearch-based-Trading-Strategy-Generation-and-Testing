#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Camarilla pivot levels (from previous 12h bar OHLC)
    prev_high_12h = np.roll(high, 8)  # 4h * 8 = 12h
    prev_low_12h = np.roll(low, 8)
    prev_close_12h = np.roll(close, 8)
    # Fill first 8 values
    prev_high_12h[:8] = high[:8]
    prev_low_12h[:8] = low[:8]
    prev_close_12h[:8] = close[:8]
    
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    r3_12h = prev_high_12h + 2 * (pivot_12h - prev_low_12h)
    s3_12h = prev_low_12h - 2 * (prev_high_12h - pivot_12h)
    
    # Daily trend: EMA34 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 30-period SMA
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.8 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(pivot_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma30[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R3(12h) with daily uptrend and volume
            if (price > r3_12h[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below S3(12h) with daily downtrend and volume
            elif (price < s3_12h[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to pivot or daily trend fails
            if (price < pivot_12h[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or daily trend fails
            if (price > pivot_12h[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals