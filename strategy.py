#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot levels from 1d combined with 12h EMA50 trend filter and volume confirmation
# Long when price touches or breaks above Camarilla R3 level AND 12h EMA50 is rising AND 4h volume > 1.3 * avg_volume(20)
# Short when price touches or breaks below Camarilla S3 level AND 12h EMA50 is falling AND 4h volume > 1.3 * avg_volume(20)
# Exit when price returns to Camarilla pivot point (PP) or opposite S1/R1 level
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels from daily timeframe provide strong intraday support/resistance
# 12h EMA50 ensures we trade with the higher timeframe trend while reducing noise
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (continuation of daily trend) and bear (mean reversion from extremes) markets

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1
    # S3 = PP - (H - L) * 1.1
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    pp_1d = typical_price
    r3_1d = pp_1d + range_1d * 1.1
    s3_1d = pp_1d - range_1d * 1.1
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 completed 12h bars for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/breaks above Camarilla R3, EMA50 rising, volume spike
            if (close[i] >= r3_1d_aligned[i] and close[i-1] < r3_1d_aligned[i-1] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches/breaks below Camarilla S3, EMA50 falling, volume spike
            elif (close[i] <= s3_1d_aligned[i] and close[i-1] > s3_1d_aligned[i-1] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point or below S1
            if close[i] <= pp_1d_aligned[i] or close[i] <= s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point or above R1
            if close[i] >= pp_1d_aligned[i] or close[i] >= r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals