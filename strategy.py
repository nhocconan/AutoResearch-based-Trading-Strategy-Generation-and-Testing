#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily high/low for pivot calculation
    daily_high = get_htf_data(prices, '1d')['high'].values
    daily_low = get_htf_data(prices, '1d')['low'].values
    daily_close = get_htf_data(prices, '1d')['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = (daily_high + daily_low + daily_close) / 3
    range_val = daily_high - daily_low
    r3 = pivot + (range_val * 1.1)
    s3 = pivot - (range_val * 1.1)
    r4 = pivot + (range_val * 1.5)
    s4 = pivot - (range_val * 1.5)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), r3)
    s3_6h = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), s3)
    r4_6h = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), r4)
    s4_6h = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), s4)
    
    # 6h volume average (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 20)  # warmup for volume and pivots
    for i in range(start, n):
        if np.isnan(avg_vol[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Fade at S3/R3: long at S3 bounce, short at R3 rejection
            if price > s3_6h[i] and price < s3_6h[i] * 1.005 and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            elif price < r3_6h[i] and price > r3_6h[i] * 0.995 and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            # Breakout continuation at S4/R4
            elif price < s4_6h[i] and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            elif price > r4_6h[i] and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 or breaks below S3 with volume
            if price >= r3_6h[i] or (price < s3_6h[i] and vol > 1.5 * avg_vol[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S3 or breaks above R3 with volume
            if price <= s3_6h[i] or (price > r3_6h[i] and vol > 1.5 * avg_vol[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_FadeBreakout"
timeframe = "6h"
leverage = 1.0