#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance in ranging markets, with weekly trend filter (EMA50) to avoid counter-trend trades, and daily volume surge to confirm institutional participation. Works in bull markets by buying pullbacks to S3 in uptrends and in bear markets by selling bounces to R3 in downtrends.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 5-period Camarilla levels (R3, S3)
    camarilla_period = 5
    camarilla_multiplier = 1.1/2  # For R3/S3 levels
    
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(camarilla_period-1, n):
        window_high = np.max(high[i-camarilla_period+1:i+1])
        window_low = np.min(low[i-camarilla_period+1:i+1])
        window_close = close[i]
        
        # Calculate pivot point
        pivot = (window_high + window_low + window_close) / 3
        
        # Calculate R3 and S3 levels
        r3[i] = pivot + camarilla_multiplier * (window_high - window_low)
        s3[i] = pivot - camarilla_multiplier * (window_high - window_low)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly
    ema_period = 50
    ema_1w = np.zeros_like(close_1w)
    alpha = 2 / (ema_period + 1)
    
    # Initialize with SMA
    ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
    for i in range(ema_period, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on daily
    vol_ma_period = 20
    vol_ma_1d = np.zeros_like(volume_1d)
    for i in range(vol_ma_period-1, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-vol_ma_period+1:i+1])
    
    # Align daily volume average to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below EMA50
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Daily volume confirmation: current 12h volume > 1.5x daily average volume
        # Convert daily average to 12h equivalent (approx: daily volume / 2)
        vol_threshold = vol_ma_1d_aligned[i] * 1.5
        vol_confirm = volume[i] > vol_threshold
        
        if position == 0:
            # LONG: Price at S3 support + weekly uptrend + volume confirmation
            if (close[i] <= s3[i] and weekly_uptrend and vol_confirm):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 resistance + weekly downtrend + volume confirmation
            elif (close[i] >= r3[i] and weekly_downtrend and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above midpoint or weekly trend changes
            midpoint = (r3[i] + s3[i]) / 2
            if (close[i] >= midpoint or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below midpoint or weekly trend changes
            midpoint = (r3[i] + s3[i]) / 2
            if (close[i] <= midpoint or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals