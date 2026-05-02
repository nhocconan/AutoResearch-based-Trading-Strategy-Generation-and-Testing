#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for HTF trend alignment to reduce whipsaw vs shorter timeframes
# Camarilla levels from 4h provide clear structure for breakouts
# Breakout at R3/S3 with volume spike confirms institutional participation
# 4h EMA50 trend filter ensures alignment with 4h trend
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag
# Discrete position sizing: 0.20 (20% of capital) to minimize fee churn while maintaining reasonable exposure
# Works in both bull and bear markets by following 4h trend

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Volume_Session"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3, S3, R4, S4
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + Range * 1.1/2
    # S3 = Close - Range * 1.1/2
    # R4 = Close + Range * 1.1
    # S4 = Close - Range * 1.1
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = close_4h + range_4h * 1.1 / 2.0
    s3_4h = close_4h - range_4h * 1.1 / 2.0
    r4_4h = close_4h + range_4h * 1.1
    s4_4h = close_4h - range_4h * 1.1
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(r4_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike AND price > 4h EMA50 (bullish trend)
            if (close[i] > r3_4h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume spike AND price < 4h EMA50 (bearish trend)
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 4h EMA50 (trend change)
            if close[i] < s3_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 4h EMA50 (trend change)
            if close[i] > r3_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals