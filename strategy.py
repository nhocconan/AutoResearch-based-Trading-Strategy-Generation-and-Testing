#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla pivot point (PP).
# Uses discrete position sizing (0.20) to limit drawdown and fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Uses 4h/1d for signal direction, 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r3 = pivot + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pivot - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, uptrend (price > 4h EMA50), volume confirmation
            if (curr_high > r3_aligned[i] and 
                curr_close > ema_50_4h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: break below S3, downtrend (price < 4h EMA50), volume confirmation
            elif (curr_low < s3_aligned[i] and 
                  curr_close < ema_50_4h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below pivot point
            if curr_close < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above pivot point
            if curr_close > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals