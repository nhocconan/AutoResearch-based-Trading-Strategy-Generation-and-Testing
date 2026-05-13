#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h HMA21 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND close > 12h HMA21 AND volume > 2.0x average
# Short when price breaks below Camarilla S3 AND close < 12h HMA21 AND volume > 2.0x average
# Exit when price crosses Camarilla pivot point (mean reversion) OR trend reversal (price crosses 12h HMA21)
# Uses 4h timeframe for optimal trade frequency, Camarilla for structure, 12h HMA for trend filter, volume spike for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.
# HMA reduces lag vs EMA/SMA, improving responsiveness in ranging markets while maintaining trend fidelity.

name = "4h_Camarilla_R3S3_Breakout_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average: reduces lag while maintaining smoothness"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculation helper
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMAs
    wma_half = wma(arr, half_period)
    wma_full = wma(arr, period)
    
    # Handle alignment
    diff = 2 * wma_half - wma_full
    # Pad diff to match original length
    pad_len = len(arr) - len(diff)
    if pad_len > 0:
        diff = np.concatenate([np.full(pad_len, np.nan), diff])
    
    # Final WMA of diff
    hma = wma(diff, sqrt_period)
    pad_len = len(arr) - len(hma)
    if pad_len > 0:
        hma = np.concatenate([np.full(pad_len, np.nan), hma])
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Camarilla levels on 4h data (using previous day's range)
    # Need previous day's high/low - we'll use rolling window of 6 bars (approx 1 day at 4h)
    if len(high_4h) >= 6:
        prev_day_high = pd.Series(high_4h).rolling(window=6, min_periods=6).max().shift(1).values
        prev_day_low = pd.Series(low_4h).rolling(window=6, min_periods=6).min().shift(1).values
        prev_day_close = pd.Series(close_4h).rolling(window=6, min_periods=6).last().shift(1).values
    else:
        prev_day_high = np.full_like(high_4h, np.nan)
        prev_day_low = np.full_like(high_4h, np.nan)
        prev_day_close = np.full_like(close_4h, np.nan)
    
    # Camarilla formulas
    R3 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 4
    S3 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 4
    PP = (prev_day_high + prev_day_low + prev_day_close) / 3  # Pivot Point
    
    # Align Camarilla levels to 4h timeframe (already aligned since calculated on 4h)
    # But we need to ensure proper alignment with look-ahead prevention
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    PP_aligned = align_htf_to_ltf(prices, df_4h, PP)
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close for trend filter
    hma21_12h = calculate_hma(close_12h, 21)
    hma21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma21_12h)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data for Camarilla and HMA
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(hma21_12h_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > R3 AND close > 12h HMA21 AND volume spike
            if close[i] > R3_aligned[i] and close[i] > hma21_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND close < 12h HMA21 AND volume spike
            elif close[i] < S3_aligned[i] and close[i] < hma21_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < PP (mean reversion) OR trend reversal (close < 12h HMA21)
            if close[i] < PP_aligned[i] or close[i] < hma21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > PP (mean reversion) OR trend reversal (close > 12h HMA21)
            if close[i] > PP_aligned[i] or close[i] > hma21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals