#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot R3/S3 Breakout with 1d Volume Spike and 12h EMA50 Trend Filter
# Buy when price breaks above Weekly R3 (resistance 3) with volume > 2x 20-bar average AND 12h EMA50 rising
# Sell when price breaks below Weekly S3 (support 3) with volume > 2x 20-bar average AND 12h EMA50 falling
# Weekly pivots calculated from prior week's OHLC: R3 = High + 2*(High - Low), S3 = Low - 2*(High - Low)
# Volume spike confirms institutional interest; 12h EMA50 trend filter ensures alignment with intermediate trend
# Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend
# Target: 12-35 trades/year via tight breakout conditions requiring volume and trend confluence

name = "6h_WeeklyPivot_R3S3_Breakout_1dVolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: R3, S3 from prior week's OHLC
    # R3 = High + 2*(High - Low), S3 = Low - 2*(High - Low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_range = high_1w - low_1w
    r3 = high_1w + 2 * weekly_range  # Resistance 3
    s3 = low_1w - 2 * weekly_range   # Support 3
    
    # Align weekly pivot levels to 6h timeframe (use prior week's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = np.nan
    ema_50_rising = ema_50_12h > ema_50_12h_prev
    ema_50_falling = ema_50_12h < ema_50_12h_prev
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 30)  # Need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above R3, volume spike, AND 12h EMA50 rising
            if close[i] > r3_aligned[i] and volume_spike_aligned[i] and ema_50_rising_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, volume spike, AND 12h EMA50 falling
            elif close[i] < s3_aligned[i] and volume_spike_aligned[i] and ema_50_falling_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price breaks below S3 or volume dries up
            if close[i] < s3_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price breaks above R3 or volume dries up
            if close[i] > r3_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals