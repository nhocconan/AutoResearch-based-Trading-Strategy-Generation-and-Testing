#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 breakout with 1d trend filter (price > EMA34) and volume spike
# Long when price breaks above 12h Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume > 3.0 * avg_volume(20) on 6h
# Short when price breaks below 12h Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume > 3.0 * avg_volume(20) on 6h
# Exit when price crosses back through the 12h Camarilla midpoint (R3/S3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R3/S3 levels provide good breakout structure with moderate false breakouts
# 1d EMA34 filter ensures we trade with the higher timeframe trend, reducing counter-trend whipsaw
# Volume spike confirmation (3.0x) validates breakout strength with high threshold to minimize overtrading
# Works in both bull and bear markets by only trading in direction of 1d trend

name = "6h_12hCamarillaR3S3_1dEMA34_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least one completed 12h bar
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_low_12h = high_12h - low_12h
    camarilla_r3_12h = close_12h + 1.1 * high_low_12h * 1.1 / 4.0
    camarilla_s3_12h = close_12h - 1.1 * high_low_12h * 1.1 / 4.0
    camarilla_mid_12h = (camarilla_r3_12h + camarilla_s3_12h) / 2.0
    
    # Align 12h Camarilla to 6h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid_12h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 3.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3, price > 1d EMA34 (uptrend), volume spike, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3, price < 1d EMA34 (downtrend), volume spike, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals