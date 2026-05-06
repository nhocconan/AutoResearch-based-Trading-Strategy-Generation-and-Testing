#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND 1w close > 1w EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below S3 AND 1w close < 1w EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the Camarilla pivot point (PP) from 1d
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R3/S3 provides strong intraday support/resistance levels
# 1w EMA34 filters for higher timeframe trend alignment
# Volume confirmation reduces false breakouts during low participation

name = "6h_Camarilla_R3S3_Breakout_1wEMA34_Volume_v1"
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
    
    # Calculate 1d Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    # We use the previous completed 1d bar to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    camarilla_pp = np.zeros(len(high_1d))
    camarilla_r3 = np.zeros(len(high_1d))
    camarilla_s3 = np.zeros(len(high_1d))
    camarilla_r4 = np.zeros(len(high_1d))
    camarilla_s4 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        if i == 0:
            # For first bar, use same values (will be aligned properly)
            camarilla_pp[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        else:
            camarilla_pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
        
        rang = high_1d[i-1] - low_1d[i-1] if i > 0 else high_1d[i] - low_1d[i]
        camarilla_r3[i] = camarilla_pp[i] + rang * 1.1 / 2
        camarilla_s3[i] = camarilla_pp[i] - rang * 1.1 / 2
        camarilla_r4[i] = camarilla_pp[i] + rang * 1.1
        camarilla_s4[i] = camarilla_pp[i] - rang * 1.1
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point (PP) from above
            if close[i] <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot point (PP) from below
            if close[i] >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals