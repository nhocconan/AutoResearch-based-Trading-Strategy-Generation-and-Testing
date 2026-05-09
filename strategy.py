#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend and 1d structure for entries.
# Uses 4h EMA50 for trend filter and 1d Camarilla S3/R3 for breakout levels.
# Designed for low trade frequency (15-35/year) to avoid fee drag in 1h timeframe.
# Works in both bull/bear markets by requiring alignment with 4h trend and 1d structure.
name = "1h_Camarilla_S3R3_4hEMA50_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla S3/R3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla pivot levels (S3 and R3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    # S3 = C - (H - L) * 1.1/2
    # R3 = C + (H - L) * 1.1/2
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 1h timeframe
    s3_1h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1h = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(s3_1h[i]) or np.isnan(r3_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above 1d R3, 4h uptrend (price > EMA50), volume breakout
            if (close[i] > r3_1h[i] and 
                close[i] > ema_50_1h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price below 1d S3, 4h downtrend (price < EMA50), volume breakdown
            elif (close[i] < s3_1h[i] and 
                  close[i] < ema_50_1h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below 1d S3 or trend reversal
            if close[i] < s3_1h[i] or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 1d R3 or trend reversal
            if close[i] > r3_1h[i] or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals