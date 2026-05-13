#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 1.5x average.
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 1.5x average.
# Exit when price retests the broken level (R3 for long, S3 for short) OR trend reversal.
# Uses 4h timeframe for optimal trade frequency, Camarilla levels for institutional structure,
# 1d EMA for trend filter, volume for confirmation. Target: 75-200 total trades over 4 years.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for 4h: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_range = prev_high_4h - prev_low_4h
    r3_4h = prev_close_4h + 1.1 * camarilla_range / 2
    s3_4h = prev_close_4h - 1.1 * camarilla_range / 2
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (1.5 * vol_ma_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 AND price > 1d EMA34 AND volume confirmation
            if close[i] > r3_4h[i] and close[i] > ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 AND price < 1d EMA34 AND volume confirmation
            elif close[i] < s3_4h[i] and close[i] < ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests R3 (break failed) OR trend reversal (price < 1d EMA34)
            if close[i] <= r3_4h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests S3 (break failed) OR trend reversal (price > 1d EMA34)
            if close[i] >= s3_4h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals