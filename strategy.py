#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction (from Camarilla) + volume spike confirmation
# Long when: price breaks above 6h Donchian(20) high AND 1d weekly pivot shows bullish bias (close > R3) AND volume > 2x 20-period MA
# Short when: price breaks below 6h Donchian(20) low AND 1d weekly pivot shows bearish bias (close < S3) AND volume > 2x 20-period MA
# Exit when: price returns to 6h Donchian(20) midpoint OR opposite breakout occurs
# Uses Donchian for structure, weekly pivot for bias, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Donchian20_1dWeeklyPivot_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 6h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 1d data ONCE before loop for weekly pivot calculation (Camarilla)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # need at least a week for weekly pivot
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels from prior week's daily OHLC
    # We'll use the prior week's high, low, close to calculate weekly pivot
    # For simplicity, we approximate weekly levels using rolling window
    if len(df_1d) >= 5:
        # Get prior week's OHLC (5 trading days)
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        
        # Calculate Camarilla levels for weekly timeframe
        # Camarilla: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
        #          S3 = close - ((high-low) * 1.1/4), S4 = close - ((high-low) * 1.1/2)
        weekly_range = weekly_high - weekly_low
        camarilla_r3 = weekly_close + (weekly_range * 1.1 / 4)
        camarilla_s3 = weekly_close - (weekly_range * 1.1 / 4)
        
        # Bullish bias: close > R3, Bearish bias: close < S3
        weekly_bullish = df_1d['close'].values > camarilla_r3
        weekly_bearish = df_1d['close'].values < camarilla_s3
    else:
        weekly_bullish = np.full(len(df_1d), False)
        weekly_bearish = np.full(len(df_1d), False)
        camarilla_r3 = np.full(len(df_1d), np.nan)
        camarilla_s3 = np.full(len(df_1d), np.nan)
    
    # Align 1d weekly pivot bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + weekly bullish + volume filter
            if (donchian_breakout_up[i] and 
                weekly_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + weekly bearish + volume filter
            elif (donchian_breakout_down[i] and 
                  weekly_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR short breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR long breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals