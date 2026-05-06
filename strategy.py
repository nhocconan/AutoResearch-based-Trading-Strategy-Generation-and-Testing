#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot breakout with 4h EMA trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 level AND 4h EMA20 > EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 level AND 4h EMA20 < EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 1d Camarilla midpoint (R3/S3 midpoint) or opposite S3/R3 level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Camarilla provides strong intraday pivot structure from daily session
# 4h EMA filter ensures alignment with intermediate trend, reducing counter-trend trades
# High volume confirmation (2.0x) filters weak breakouts, reducing false signals
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "4h_1dCamarilla_R3S3_Breakout_4hEMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R3 = Pivot + Range * 1.1 / 2
    # S3 = Pivot - Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    camarilla_s3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    camarilla_mid_1d = (camarilla_r3_1d + camarilla_s3_1d) / 2.0  # midpoint between R3 and S3
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 and EMA50
    close_series_4h = pd.Series(close_4h)
    ema_20_4h = close_series_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA values to 4h timeframe (no additional delay needed for EMA)
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 4h EMA20 > EMA50 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_20_aligned[i] > ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 4h EMA20 < EMA50 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_20_aligned[i] < ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Camarilla midpoint or S3 level (reversal or profit take)
            if close[i] <= camarilla_mid_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Camarilla midpoint or R3 level (reversal or profit take)
            if close[i] >= camarilla_mid_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals