#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction.
- Camarilla levels: Calculated from previous 1d OHLC (R1, S1, R3, S3, etc.).
- Trend filter: 12h EMA50 slope > 0 for longs, < 0 for shorts (using 3-bar slope to avoid noise).
- Volume confirmation: 4h volume > 1.5 * 20-period average volume.
- Entry: Long when price > R1 AND uptrend AND volume confirmation.
         Short when price < S1 AND downtrend AND volume confirmation.
- Exit: Opposite Camarilla break (price < R1 for long exit, price > S1 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
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
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate 3-bar slope to determine trend direction
    ema_slope = np.zeros_like(ema_50)
    ema_slope[3:] = (ema_50[3:] - ema_50[:-3]) / 3  # 3-period slope
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Calculate Camarilla levels from previous 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1, S1, R3, S3
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    r3 = close_1d + 1.1 * rng / 4
    s3 = close_1d - 1.1 * rng / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_slope_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 slope > 0 for uptrend, < 0 for downtrend
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: price < R1
            if position == 1:
                if curr_close < r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S1
            elif position == -1:
                if curr_close > s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > R1 AND uptrend AND volume confirmation
            long_condition = (curr_close > r1_aligned[i] and 
                            uptrend and
                            volume_confirm)
            
            # Short: price < S1 AND downtrend AND volume confirmation
            short_condition = (curr_close < s1_aligned[i] and 
                             downtrend and
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

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0