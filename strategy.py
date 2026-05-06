#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above weekly R3 AND 1d EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly S3 AND 1d EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Exit when price touches weekly pivot point or opposite Camarilla level (R3/S3)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong structural levels aligned with weekly session
# 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "6h_1wCamarilla_R3S3_Breakout_1dEMATrend_Volume"
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
    
    # Get 1w data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need sufficient data for weekly pivot
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + 1.1 * Range / 2, S3 = Pivot - 1.1 * Range / 2
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    camarilla_r3_1w = pivot_1w + 1.1 * range_1w / 2.0
    camarilla_s3_1w = pivot_1w - 1.1 * range_1w / 2.0
    camarilla_pivot_1w = pivot_1w  # Midpoint for exit
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot_1w)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches weekly pivot or S3 (reversal or profit take)
            if close[i] <= camarilla_pivot_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches weekly pivot or R3 (reversal or profit take)
            if close[i] >= camarilla_pivot_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals