#!/usr/bin/env python3
# 12h_camarilla_volatility_breakout_v2
# Hypothesis: Uses daily Camarilla pivot levels on 12h timeframe. Enters on break of R3/S3 with volume confirmation and ATR-based volatility filter. Uses 1d trend filter to avoid counter-trend trades. Designed for 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_volatility_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (focus on R3/S3 for entries)
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 12h timeframe (using previous day's levels)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d trend filter (EMA25)
    if len(df_1d) < 25:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    alpha_1d = 2 / (25 + 1)
    ema25_1d = np.zeros(len(df_1d))
    ema25_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema25_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema25_1d[i-1]
    
    # 1d trend: 1 if close > EMA25, -1 if close < EMA25
    trend_1d = np.where(close_1d > ema25_1d, 1, -1)
    trend_12h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volatility filter: ATR(14) ratio (current ATR / 50-period ATR mean)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar
    atr14 = np.zeros(n)
    atr_sum = 0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i-14]
        if i >= 13:
            atr14[i] = atr_sum / 14
    
    atr50 = np.zeros(n)
    atr_sum50 = 0
    for i in range(n):
        atr_sum50 += tr[i]
        if i >= 50:
            atr_sum50 -= tr[i-50]
        if i >= 49:
            atr50[i] = atr_sum50 / 50
    
    # Avoid division by zero
    atr_ratio = np.zeros(n)
    for i in range(n):
        if atr50[i] > 0:
            atr_ratio[i] = atr14[i] / atr50[i]
        else:
            atr_ratio[i] = 1.0
    
    # Volume filter: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(trend_12h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (1.5x average)
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Volatility filter: trade only when volatility is elevated (ATR ratio > 0.8) but not extreme (< 3.0)
        vol_filter_ok = (atr_ratio[i] > 0.8) and (atr_ratio[i] < 3.0)
        
        if position == 1:  # Long position
            # Exit: close below R3 or trend turns bearish or volatility drops
            if (close[i] < r3_12h[i] or 
                trend_12h[i] == -1 or 
                not vol_filter_ok):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above S3 or trend turns bullish or volatility drops
            if (close[i] > s3_12h[i] or 
                trend_12h[i] == 1 or 
                not vol_filter_ok):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above R3 with volume, bullish trend, and adequate volatility
            if (close[i] > r3_12h[i] and 
                vol_ok and 
                trend_12h[i] == 1 and
                vol_filter_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: break below S3 with volume, bearish trend, and adequate volatility
            elif (close[i] < s3_12h[i] and 
                  vol_ok and 
                  trend_12h[i] == -1 and
                  vol_filter_ok):
                position = -1
                signals[i] = -0.25
    
    return signals