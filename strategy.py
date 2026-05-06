#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 level AND 1w EMA50 is rising AND 6h volume > 1.5 * avg_volume(24)
# Short when price breaks below 1d Camarilla S3 level AND 1w EMA50 is falling AND 6h volume > 1.5 * avg_volume(24)
# Exit when price returns to 1d Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Camarilla provides key intraday structure from prior session
# 1w EMA50 ensures we trade with the weekly trend while reducing whipsaws
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "6h_1dCamarillaR3S3_Breakout_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # PP = (H + L + C) / 3
    # R3 = PP + 2 * (H - L)
    # S3 = PP - 2 * (H - L)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + 2.0 * (high_1d - low_1d)
    s3_1d = pp_1d - 2.0 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 completed weekly bars for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 24-period average volume on 6h (approx 6d)
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(avg_volume_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, EMA50 rising, volume spike
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, EMA50 falling, volume spike
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot point
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot point
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals