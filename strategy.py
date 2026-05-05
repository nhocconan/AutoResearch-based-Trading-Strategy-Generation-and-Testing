#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d Camarilla pivot (R3/S3) touch + volume confirmation
# Long when: price touches 1d Camarilla S3 AND breaks above 12h Donchian(20) high AND volume > 1.5x 20-period MA
# Short when: price touches 1d Camarilla R3 AND breaks below 12h Donchian(20) low AND volume > 1.5x 20-period MA
# Exit when: price reaches 12h Donchian midpoint OR opposite breakout occurs
# Uses Camarilla for mean reversion bias, Donchian for breakout structure, volume for conviction
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
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # need sufficient data for Camarilla
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from prior day's OHLC
    if len(df_1d) >= 2:
        # Get prior day's OHLC (yesterday's high, low, close)
        daily_high = df_1d['high'].shift(1).values
        daily_low = df_1d['low'].shift(1).values
        daily_close = df_1d['close'].shift(1).values
        
        # Calculate Camarilla levels for daily timeframe
        # Camarilla: R3 = close + ((high-low) * 1.1/4), S3 = close - ((high-low) * 1.1/4)
        daily_range = daily_high - daily_low
        camarilla_r3 = daily_close + (daily_range * 1.1 / 4)
        camarilla_s3 = daily_close - (daily_range * 1.1 / 4)
        
        # Bullish bias: close <= S3 (touch or below), Bearish bias: close >= R3 (touch or above)
        daily_bullish = df_1d['close'].values <= camarilla_s3
        daily_bearish = df_1d['close'].values >= camarilla_r3
    else:
        daily_bullish = np.full(len(df_1d), False)
        daily_bearish = np.full(len(df_1d), False)
        camarilla_r3 = np.full(len(df_1d), np.nan)
        camarilla_s3 = np.full(len(df_1d), np.nan)
    
    # Align 1d Camarilla touch bias to 12h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches S3 + Donchian breakout up + volume filter
            if (daily_bullish_aligned[i] == 1.0 and 
                donchian_breakout_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price touches R3 + Donchian breakout down + volume filter
            elif (daily_bearish_aligned[i] == 1.0 and 
                  donchian_breakout_down[i] and 
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