#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: Camarilla pivot levels from 1-day timeframe act as strong support/resistance. 
# Enter long at S3 with volume confirmation when price closes above S3, short at R3 when price closes below R3.
# Uses 1-day trend filter (price above/below EMA50) to align with higher timeframe direction.
# Designed for low trade frequency (~20-40/year) to minimize fee drag on 12h timeframe.
# Works in bull/bear by following 1-day trend while using intraday pivot reversals for entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # PP = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.1 / 2)
    # S2 = C - (Range * 1.1 / 4)
    # S1 = C - (Range * 1.1 / 6)
    # R1 = C + (Range * 1.1 / 6)
    # R2 = C + (Range * 1.1 / 4)
    # R3 = C + (Range * 1.1 / 2)
    pp = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    s3 = close_1d - (rang * 1.1 / 2.0)
    r3 = close_1d + (rang * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h (these levels are fixed for the entire day)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S3 (broken support) OR price < EMA50 (trend change)
            if (close[i] < s3_12h[i]) or (close[i] < ema_50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R3 (broken resistance) OR price > EMA50 (trend change)
            if (close[i] > r3_12h[i]) or (close[i] > ema_50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above S3 + volume + price > EMA50 (uptrend)
            if (close[i] > s3_12h[i]) and volume_filter[i] and (close[i] > ema_50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below R3 + volume + price < EMA50 (downtrend)
            elif (close[i] < r3_12h[i]) and volume_filter[i] and (close[i] < ema_50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals