#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Camarilla R3/S3 levels from 1d with 12h EMA trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 12h EMA34 > EMA89 AND volume > 1.3 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 12h EMA34 < EMA89 AND volume > 1.3 * avg_volume(20)
# Exit when price touches 1d Camarilla pivot point or opposite S3/R3 level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R3/S3 provides strong intraday reversal/breakout levels
# 12h EMA filter ensures alignment with medium-term trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts at R4/S4) and bear (mean reversion at R3/S3)

name = "6h_1dCamarilla_R3S3_12hEMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need sufficient data for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R3 = pivot + (range * 1.1/2)
    # S3 = pivot - (range * 1.1/2)
    # R4 = pivot + (range * 1.1)
    # S4 = pivot - (range * 1.1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    camarilla_s3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    camarilla_r4_1d = pivot_1d + (range_1d * 1.1)
    camarilla_s4_1d = pivot_1d - (range_1d * 1.1)
    camarilla_pivot_1d = pivot_1d
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA89
    close_series_12h = pd.Series(close_12h)
    ema_34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_12h = close_series_12h.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMA values to 6h timeframe (wait for completed 12h bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_89_aligned = align_htf_to_ltf(prices, df_12h, ema_89_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 12h EMA34 > EMA89 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 12h EMA34 < EMA89 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Camarilla pivot or S4 (profit take or reversal)
            if close[i] <= camarilla_pivot_aligned[i] or close[i] <= camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Camarilla pivot or R4 (profit take or reversal)
            if close[i] >= camarilla_pivot_aligned[i] or close[i] >= camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals