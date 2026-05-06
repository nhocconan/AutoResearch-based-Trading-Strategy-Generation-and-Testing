#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakouts with daily trend filter and volume confirmation
# Weekly Camarilla levels (R3/S3) from prior week provide strong support/resistance
# Daily EMA34 trend filter ensures breakouts align with intermediate trend
# Volume confirmation (>2.0x 24-bar average) filters weak breakouts
# Discrete sizing 0.25 to balance return and drawdown; target 80-120 total trades over 4 years (20-30/year)
# Works in bull markets via breakout continuation and bear markets via mean reversion at extreme levels

name = "6h_WeeklyCamarilla_R3S3_DailyEMA34_VolumeConfirm_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly Camarilla pivot levels (using prior weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_high_1w = np.full(len(close_1w), np.nan)
    camarilla_low_1w = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        h = high_1w[i-1]
        l = low_1w[i-1]
        c = close_1w[i-1]
        camarilla_high_1w[i] = c + ((h - l) * 1.1 / 4)  # R3
        camarilla_low_1w[i] = c - ((h - l) * 1.1 / 4)   # S3
    
    # Daily EMA34 trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation filter (>2.0x 24-bar average = 4 days on 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    # Align HTF indicators to 6h timeframe
    camarilla_high_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high_1w)
    camarilla_low_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low_1w)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_high_1w_aligned[i]) or np.isnan(camarilla_low_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Weekly R3 AND price > Daily EMA34 AND volume spike
            if close[i] > camarilla_high_1w_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Weekly S3 AND price < Daily EMA34 AND volume spike
            elif close[i] < camarilla_low_1w_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests Weekly S3 from above (mean reversion)
            if close[i] <= camarilla_low_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests Weekly R3 from below (mean reversion)
            if close[i] >= camarilla_high_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals