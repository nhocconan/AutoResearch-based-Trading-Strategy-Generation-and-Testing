#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Camarilla pivot levels from 1d with volume confirmation and 1d trend filter.
# Camarilla pivot levels provide clear support/resistance based on prior day's range.
# Long when price breaks above R3 with volume confirmation and bullish 1d trend.
# Short when price breaks below S3 with volume confirmation and bearish 1d trend.
# Exit when price returns to the central pivot point (PP).
# Designed for ~15-25 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Handle first day (no previous day)
    high_prev[0] = high_prev[1] if len(high_prev) > 1 else high_prev[0]
    low_prev[0] = low_prev[1] if len(low_prev) > 1 else low_prev[0]
    close_prev[0] = close_prev[1] if len(close_prev) > 1 else close_prev[0]
    
    # Calculate pivot point and Camarilla levels
    pp = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels
    r3 = pp + (range_val * 1.1 / 2.0)  # R3 = PP + 1.1 * (H-L)/2
    s3 = pp - (range_val * 1.1 / 2.0)  # S3 = PP - 1.1 * (H-L)/2
    
    # Align 1d Camarilla levels to execution timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: volume > 1.5x 20-period average (execution timeframe)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d Camarilla (1), volume MA (20), 1d EMA (50)
    start_idx = max(1, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and bullish trend
            if price > r3_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume and bearish trend
            elif price < s3_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0