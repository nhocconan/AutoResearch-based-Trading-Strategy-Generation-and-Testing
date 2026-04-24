#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction and volume spike filter.
- Camarilla levels: Calculated from previous 1d OHLC (R1, S1, R3, S3, etc.)
- Trend filter: 12h EMA50 slope > 0 for longs, < 0 for shorts.
- Volume confirmation: 4h volume > 2.0 * 20-period average volume.
- Entry: Long when price > R1 AND 12h EMA50 rising AND volume spike.
         Short when price < S1 AND 12h EMA50 falling AND volume spike.
- Exit: Opposite Camarilla break (price < R1 for long exit, price > S1 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in the direction of 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA50 calculation
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA50 slope for trend direction (positive = rising, negative = falling)
    ema50_slope = np.diff(ema50, prepend=ema50[0])
    
    # Align EMA50 slope to 4h timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Calculate 12h volume average for spike confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's values (available at close of previous day)
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate Camarilla levels
    R1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    S1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    R3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    S3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_slope_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 slope direction
        rising_trend = ema50_slope_aligned[i] > 0
        falling_trend = ema50_slope_aligned[i] < 0
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: price < R1
            if position == 1:
                if curr_close < R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S1
            elif position == -1:
                if curr_close > S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > R1 AND rising trend AND volume confirmation
            long_condition = (curr_close > R1_aligned[i] and 
                            rising_trend and
                            volume_confirm)
            
            # Short: price < S1 AND falling trend AND volume confirmation
            short_condition = (curr_close < S1_aligned[i] and 
                             falling_trend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0