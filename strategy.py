#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Supertrend trend filter + 1w Camarilla pivot breakout + volume confirmation.
Long when price breaks above weekly Camarilla R3 level with 1d Supertrend bullish and volume > 1.5x 20-period 1d volume average.
Short when price breaks below weekly Camarilla S3 level with 1d Supertrend bearish and volume > 1.5x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Weekly pivots provide structural levels; Supertrend filters for trending markets only; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
"""

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
    
    # Get 1d data for Supertrend and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # first value is simple average
            result[period-1] = np.nanmean(data[:period])
            # subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, atr_period)
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.full_like(close_1d, np.nan)
    final_lb = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(basic_ub[i]) or np.isnan(basic_lb[i]):
            continue
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            # Final Upper Band
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
            
            # Final Lower Band
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            continue
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_1d[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_1d[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_1d[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_1d[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Supertrend trend direction: 1 = bullish (price above supertrend), -1 = bearish (price below supertrend)
    supertrend_dir = np.where(close_1d > supertrend, 1, -1)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Camarilla levels
    # Camarilla: R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.1 / 4)
    camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.1 / 4)
    
    # Align all to 12h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Supertrend and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R3 with bullish Supertrend and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                supertrend_dir_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S3 with bearish Supertrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  supertrend_dir_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Camarilla R2 level
            camarilla_r2 = close_1w + ((high_1w - low_1w) * 1.1/6)
            camarilla_r2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
            if close[i] < camarilla_r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Camarilla S2 level
            camarilla_s2 = close_1w - ((high_1w - low_1w) * 1.1/6)
            camarilla_s2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
            if close[i] > camarilla_s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dSupertrend_1wCamarilla_S3R3_Volume_Confirm"
timeframe = "12h"
leverage = 1.0