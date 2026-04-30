#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous hour for structure
# Only trade breakouts above R3 or below S3 in direction of 4h EMA50 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop (MTF Rule #1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # warmup for Camarilla calculation (need previous hour)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Need previous hour's high/low/close for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Previous hour's high/low/close for Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 4.0)  # R3 level
        s3 = pivot - (range_val * 1.1 / 4.0)  # S3 level
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and session
            if volume_spike:
                # Bullish entry: price breaks above R3 AND above 4h EMA50 (uptrend)
                if curr_close > r3 and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below S3 AND below 4h EMA50 (downtrend)
                elif curr_close < s3 and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 4h EMA50
            if curr_close < s3 or curr_close < curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 4h EMA50
            if curr_close > r3 or curr_close > curr_ema_50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals