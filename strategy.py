#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d Camarilla R3/S3 touch + volume spike confirmation
# Long when: price touches 1d Camarilla S3 support AND 12h Donchian(20) shows bullish bias (close > midpoint) AND volume > 1.5x 20-period MA
# Short when: price touches 1d Camarilla R3 resistance AND 12h Donchian(20) shows bearish bias (close < midpoint) AND volume > 1.5x 20-period MA
# Exit when: price reaches 12h Donchian(20) midpoint OR opposite Camarilla touch occurs
# Uses Camarilla for precise reversal levels, Donchian for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dCamarilla_R3S3_Touch_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 12h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # need sufficient data for Camarilla
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC (standard daily Camarilla)
    if len(df_1d) >= 2:
        # Use prior day's high, low, close to calculate today's Camarilla levels
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate Camarilla levels
        # R3 = close + ((high-low) * 1.1/4), S3 = close - ((high-low) * 1.1/4)
        prev_range = prev_high - prev_low
        camarilla_r3 = prev_close + (prev_range * 1.1 / 4)
        camarilla_s3 = prev_close - (prev_range * 1.1 / 4)
    else:
        camarilla_r3 = np.full(len(df_1d), np.nan)
        camarilla_s3 = np.full(len(df_1d), np.nan)
        prev_close = np.full(len(df_1d), np.nan)
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate Camarilla touch conditions (price touches S3 for long, R3 for short)
    # Allow small tolerance for touch (0.1% of price)
    touch_tolerance = 0.001
    camarilla_s3_touch = np.abs(low - camarilla_s3_aligned) <= (touch_tolerance * camarilla_s3_aligned)
    camarilla_r3_touch = np.abs(high - camarilla_r3_aligned) <= (touch_tolerance * camarilla_r3_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla S3 touch + Donchian bullish bias + volume filter
            if (camarilla_s3_touch[i] and 
                close[i] > donchian_mid[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Camarilla R3 touch + Donchian bearish bias + volume filter
            elif (camarilla_r3_touch[i] and 
                  close[i] < donchian_mid[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches Donchian midpoint OR Camarilla R3 touch (short signal)
            if (np.abs(close[i] - donchian_mid[i]) <= (0.001 * donchian_mid[i]) or camarilla_r3_touch[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches Donchian midpoint OR Camarilla S3 touch (long signal)
            if (np.abs(close[i] - donchian_mid[i]) <= (0.001 * donchian_mid[i]) or camarilla_s3_touch[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals