#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (using 5-day lookback)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using prior 5-day high/low/close (approximate weekly)
    # Use 5-day rolling window to get weekly high/low/close
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Pivot point: (H + L + C) / 3
    pp = (high_5d + low_5d + close_5d) / 3
    # Weekly resistance and support levels (Camarilla style)
    r3 = pp + (high_5d - low_5d) * 1.1
    s3 = pp - (high_5d - low_5d) * 1.1
    r4 = pp + (high_5d - low_5d) * 1.5
    s4 = pp - (high_5d - low_5d) * 1.5
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for trend filter (weekly EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 5)  # 20 for volume, 5 for weekly pivot
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume AND above weekly EMA (bullish bias)
            if price > s3_aligned[i] and vol > 1.5 * avg_vol[i] and price > ema_20_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below R3 with volume AND below weekly EMA (bearish bias)
            elif price < r3_aligned[i] and vol > 1.5 * avg_vol[i] and price < ema_20_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S4 (strong support break) or above R4 (take profit)
            if price < s4_aligned[i] or price > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R4 (strong resistance break) or below S4 (take profit)
            if price > r4_aligned[i] or price < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Weekly_Pivot_Volume_EMA_Filter"
timeframe = "6h"
leverage = 1.0