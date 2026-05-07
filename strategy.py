#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels for previous day
    # Using previous day's OHLC (already available in df_1d)
    # Camarilla R3, R4, S3, S4
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    # S4 = close - (high - low) * 1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 2
    r3 = close_1d + camarilla_width
    s3 = close_1d - camarilla_width
    r4 = close_1d + camarilla_width * 2
    s4 = close_1d - camarilla_width * 2
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1h volume average (20-period)
    vol_ma20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 (uptrend), breaks R4 with volume
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > r4_aligned[i] and 
                volume[i] > vol_ma20_1h[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price below 1d EMA34 (downtrend), breaks S4 with volume
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < s4_aligned[i] and 
                  volume[i] > vol_ma20_1h[i] * 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 (mean reversion)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above S3 (mean reversion)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals