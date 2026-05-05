#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla R3 AND 1d EMA50 > previous 1d EMA50 (uptrend) AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below 1w Camarilla S3 AND 1d EMA50 < previous 1d EMA50 (downtrend) AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses back through the 1w Camarilla midpoint (R3/S3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Camarilla provides structure that works in both bull and bear markets
# 1d EMA50 filter ensures we trade with the intermediate trend, reducing whipsaw
# Volume confirmation (1.5x) validates breakout strength without being too restrictive

name = "1d_1wCamarillaR3S3_1dEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_low_1w = high_1w - low_1w
    camarilla_r3_1w = close_1w + 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_s3_1w = close_1w - 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_mid_1w = (camarilla_r3_1w + camarilla_s3_1w) / 2.0
    
    # Align 1w Camarilla to 1d timeframe (wait for completed weekly bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1w, camarilla_mid_1w)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R3, 1d EMA50 rising (uptrend), volume confirmation, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S3, 1d EMA50 falling (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1w Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals