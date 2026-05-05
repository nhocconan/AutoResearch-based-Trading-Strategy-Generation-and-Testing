#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot breakouts with 6h volume confirmation and 1w trend filter
# Long when price breaks above 1d Camarilla R3 AND 1w EMA50 > EMA200 (bullish trend) AND volume > 1.5 * avg_volume(20) on 6h
# Short when price breaks below 1d Camarilla S3 AND 1w EMA50 < EMA200 (bearish trend) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses back through the 1d Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-160 total trades over 4 years (20-40/year) for 6h timeframe
# Camarilla R3/S3 levels provide meaningful breakout structure that reduces whipsaw
# 1w EMA50/EMA200 filter ensures we trade with the higher timeframe trend, reducing counter-trend losses
# Volume confirmation (1.5x) validates breakout strength while avoiding overtrading

name = "6h_1dCamarillaR3S3_1wEMA50EMA200_VolumeConfirm"
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
    if len(df_1d) < 2:  # Need at least one completed 1d bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, midpoint)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_low_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_l3_1d = close_1d - 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_mid_1d = (camarilla_h3_1d + camarilla_l3_1d) / 2.0
    
    # Align 1d Camarilla to 6h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1w data ONCE before loop for EMA50/EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need at least 200 completed weekly bars for EMA200
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 and EMA200
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 6h timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla H3, 1w EMA50 > EMA200 (bullish trend), volume confirmation, in session
            if (close[i] > camarilla_h3_aligned[i] and 
                ema50_1w_aligned[i] > ema200_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla L3, 1w EMA50 < EMA200 (bearish trend), volume confirmation, in session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  ema50_1w_aligned[i] < ema200_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals