#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_Trend_Volume
# Hypothesis: Use daily Camarilla pivot levels (S3, S4, R3, R4) on 12h timeframe.
# Enter long when price breaks above R3 with volume > 1.5x 20-period average and price above 12h EMA50.
# Enter short when price breaks below S3 with volume > 1.5x 20-period average and price below 12h EMA50.
# Exit when price crosses the 12h EMA50 or reverses past S4/R4.
# Uses Camarilla for intraday support/resistance, EMA50 for trend filter, volume for confirmation.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong intraday moves.

name = "12h_Camarilla_Pivot_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (based on previous day)
    # S1 = C - (H - L) * 1.05 / 6
    # S2 = C - (H - L) * 1.10 / 2
    # S3 = C - (H - L) * 1.25 / 2
    # S4 = C - (H - L) * 1.50 / 2
    # R3 = C + (H - L) * 1.25 / 2
    # R4 = C + (H - L) * 1.50 / 2
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    rang = prev_high - prev_low
    s3 = prev_close - rang * 1.25 / 2
    s4 = prev_close - rang * 1.50 / 2
    r3 = prev_close + rang * 1.25 / 2
    r4 = prev_close + rang * 1.50 / 2
    
    # Align Camarilla levels to 12h timeframe
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(r4_12h[i]) or
            np.isnan(ema_50_12h[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and above EMA50
            if close[i] > r3_12h[i] and volume[i] > vol_threshold[i] and close[i] > ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and below EMA50
            elif close[i] < s3_12h[i] and volume[i] > vol_threshold[i] and close[i] < ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA50 or breaks below S4 (strong reversal)
            if close[i] < ema_50_12h[i] or close[i] < s4_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EMA50 or breaks above R4 (strong reversal)
            if close[i] > ema_50_12h[i] or close[i] > r4_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals