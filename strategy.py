#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 level AND 1d EMA34 is rising AND volume > 1.5 * avg_volume(20) on 1h
# Short when price breaks below 4h Camarilla S3 level AND 1d EMA34 is falling AND volume > 1.5 * avg_volume(20) on 1h
# Exit when price crosses the 4h Camarilla pivot point (midpoint of R3/S3)
# Uses discrete sizing 0.20 to control risk and minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Camarilla R3/S3 provides structure from higher timeframe, reducing false signals
# 1d EMA34 ensures we trade with the dominant trend while reducing noise
# Volume confirmation (1.5x) filters breakouts with genuine participation
# Session filter (08-20 UTC) avoids low-liquidity periods
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 1d trend

name = "1h_4hCamarilla_R3S3_Breakout_1dEMA34_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:  # Need at least 5 completed 4h bars for pivot calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (R3, S3, and pivot point)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4.0
    s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4.0
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R3, 1d EMA34 rising, volume spike, in session
            if (close[i] > r3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S3, 1d EMA34 falling, volume spike, in session
            elif (close[i] < s3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 4h Camarilla pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above the 4h Camarilla pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals