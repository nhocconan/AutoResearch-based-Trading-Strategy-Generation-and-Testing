#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using weekly pivot levels for breakout/fade logic with 1d EMA34 trend filter and volume confirmation
# - Long breakout when price breaks above weekly R4 with volume expansion and price above 1d EMA34
# - Short breakdown when price breaks below weekly S4 with volume expansion and price below 1d EMA34
# - Long fade when price rejects at weekly S3 with volume exhaustion and price above 1d EMA34
# - Short fade when price rejects at weekly R3 with volume exhaustion and price below 1d EMA34
# - Exit when price crosses back below/above 1d EMA34
# - Volume expansion: current volume > 1.5x 20-period average
# - Volume exhaustion: current volume < 0.7x 20-period average
# - Designed to capture strong trends via breakouts and mean reversion via pivot rejections
# - Weekly pivots provide institutional levels that work in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_R3S3_R4S4_BreakoutFade_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculations (standard formula)
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    r4 = r3 + (high_1w - low_1w)
    s4 = s3 - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * vol_ma_20)  # Volume expansion for breakouts
    volume_exhaustion = volume < (0.7 * vol_ma_20)  # Volume exhaustion for fades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_expansion[i]) or 
            np.isnan(volume_exhaustion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R4 with volume expansion and above EMA34
            if close[i] > r4_6h[i] and volume_expansion[i] and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S4 with volume expansion and below EMA34
            elif close[i] < s4_6h[i] and volume_expansion[i] and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
            # Long fade: price rejects at weekly S3 with volume exhaustion and above EMA34
            elif low[i] <= s3_6h[i] and close[i] > s3_6h[i] and volume_exhaustion[i] and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short fade: price rejects at weekly R3 with volume exhaustion and below EMA34
            elif high[i] >= r3_6h[i] and close[i] < r3_6h[i] and volume_exhaustion[i] and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals