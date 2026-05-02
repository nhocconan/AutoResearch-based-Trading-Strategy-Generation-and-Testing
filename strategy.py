#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Uses 4h Camarilla levels for institutional S/R and 1d EMA50 for higher timeframe trend.
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3 = H + 1.1*(L-H)/6, S3 = L - 1.1*(H-L)/6
    camarilla_range = high_4h - low_4h
    r3_4h = high_4h + 1.1 * camarilla_range / 6.0
    s3_4h = low_4h - 1.1 * camarilla_range / 6.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Camarilla and 1d EMA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > r3_4h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR below 1d EMA50 (trend change)
            if close[i] < s3_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR above 1d EMA50 (trend change)
            if close[i] > r3_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals