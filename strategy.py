#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d Camarilla R3/S3 level touch + volume confirmation
# Long when: price touches 1d Camarilla S3 (support) AND breaks above 4h Donchian(20) high AND volume > 1.5x 20-period MA
# Short when: price touches 1d Camarilla R3 (resistance) AND breaks below 4h Donchian(20) low AND volume > 1.5x 20-period MA
# Exit when: price reaches 4h Donchian(20) midpoint OR opposite Camarilla level is touched with breakout
# Uses Camarilla for institutional levels, Donchian for structure, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

name = "4h_Donchian20_1dCamarilla_R3S3_Touch_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 4h
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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # need sufficient data for Camarilla calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d timeframe using prior day's OHLC
    if len(df_1d) >= 2:
        # Prior day's high, low, close
        prior_high = df_1d['high'].shift(1).values
        prior_low = df_1d['low'].shift(1).values
        prior_close = df_1d['close'].shift(1).values
        
        # Calculate Camarilla levels
        # Camarilla: R3 = close + ((high-low) * 1.1/4), S3 = close - ((high-low) * 1.1/4)
        prior_range = prior_high - prior_low
        camarilla_r3 = prior_close + (prior_range * 1.1 / 4)
        camarilla_s3 = prior_close - (prior_range * 1.1 / 4)
        
        # Price touching Camarilla levels (within 0.2% tolerance)
        touch_r3 = np.abs(close - camarilla_r3) < (0.002 * close)
        touch_s3 = np.abs(close - camarilla_s3) < (0.002 * close)
    else:
        touch_r3 = np.zeros(n, dtype=bool)
        touch_s3 = np.zeros(n, dtype=bool)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Align 1d Camarilla touch signals to 4h timeframe
    touch_r3_aligned = align_htf_to_ltf(prices, df_1d, touch_r3.astype(float))
    touch_s3_aligned = align_htf_to_ltf(prices, df_1d, touch_s3.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(touch_r3_aligned[i]) or np.isnan(touch_s3_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: touch S3 + Donchian breakout up + volume filter
            if (touch_s3_aligned[i] == 1.0 and 
                donchian_breakout_up[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: touch R3 + Donchian breakout down + volume filter
            elif (touch_r3_aligned[i] == 1.0 and 
                  donchian_breakout_down[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR touch R3 with breakout down
            if (donchian_revert_mid[i] or 
                (touch_r3_aligned[i] == 1.0 and donchian_breakout_down[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR touch S3 with breakout up
            if (donchian_revert_mid[i] or 
                (touch_s3_aligned[i] == 1.0 and donchian_breakout_up[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals