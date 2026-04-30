#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3, price > 4h EMA50, volume > 1.5x 20-bar avg, and UTC hour 8-20.
# Short when price breaks below S3, price < 4h EMA50, volume > 1.5x 20-bar avg, and UTC hour 8-20.
# Exit when price reverts to the Camarilla pivot point.
# Uses 4h EMA50 for trend alignment and session filter to reduce noise. Targets 15-35 trades/year on 1h.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (UTC) from datetime index
    session_hours = prices.index.hour  # already datetime64[ms], .hour works
    
    # Load 4h data ONCE before loop for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous 4h bar (using same 4h data)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r3 = close_4h + (range_4h * 1.1 / 4)
    s3 = close_4h - (range_4h * 1.1 / 4)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = session_hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, price > 4h EMA50, volume spike, in session
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, price < 4h EMA50, volume spike, in session
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals