#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX > 25 trend filter and volume confirmation (>1.5x avg volume).
# Uses ATR(20) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# ADX trend filter ensures we only trade with the higher timeframe trend, reducing counter-trend whipsaw.
# Camarilla R3/S3 levels provide stronger breakout/breakdown points than R1/S1, reducing false signals.
# Volume confirmation (>1.5x) ensures breakouts have institutional participation.
# Works in bull markets via trend-following breakouts and in bear markets via shorting breakdowns with trend filter.

name = "4h_Camarilla_R3_S3_Breakout_1dADXTrend_VolumeSpike_ATRStop_v1"
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
    
    # Calculate ATR(20) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to calculate today's levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        camarilla_r3[i] = prev_close + ((prev_high - prev_low) * 1.1 / 4)
        camarilla_s3[i] = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d ADX(14) for trend filter
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe (wait for daily bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND 1d ADX > 25 AND volume > 1.5x average
            if (close[i] > camarilla_r3_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S3 AND 1d ADX > 25 AND volume > 1.5x average
            elif (close[i] < camarilla_s3_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals