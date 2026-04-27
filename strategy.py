#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average and price > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x average and price < 1d EMA34.
# Exit when price crosses back below R3 (long) or above S3 (short).
# Uses 12h for pivot levels (structure), 6h for entry timing, 1d for trend filter.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (using previous day's HLC)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + 1.1 * Range * 1.1 / 2
    # S3 = Pivot - 1.1 * Range * 1.1 / 2
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + 1.1 * range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - 1.1 * range_12h * 1.1 / 2.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 24-period volume MA
    start_idx = 23
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Breakout conditions
        bullish_breakout = price > r3_aligned[i]
        bearish_breakout = price < s3_aligned[i]
        
        # Trend filter from 1d EMA34
        bullish_trend = price > ema34_aligned[i]
        bearish_trend = price < ema34_aligned[i]
        
        if position == 0:
            # Long: bullish breakout with volume and bullish trend
            if bullish_breakout and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish breakout with volume and bearish trend
            elif bearish_breakout and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below R3
            if price < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above S3
            if price > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0