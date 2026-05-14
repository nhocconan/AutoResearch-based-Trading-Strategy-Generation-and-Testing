#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter (HMA21) and volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1w HMA21 (bullish trend) AND volume > 1.5x 20-period average.
# Short when price breaks below S3 AND close < 1w HMA21 (bearish trend) AND volume > 1.5x 20-period average.
# Exit when price retests the 1w HMA21 level (mean reversion to trend) or opposite Camarilla level touched.
# Uses 1w HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Camarilla pivot levels provide high-probability reversal/breakout levels, effective in both bull and bear markets when combined with HTF trend filter.

name = "1d_Camarilla_R3S3_Breakout_1wHMA21_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / (weights.sum())
    
    wma_half = wma(arr, half_period)
    wma_full = wma(arr, period)
    
    if len(wma_half) < half_period or len(wma_full) < period:
        return np.full_like(arr, np.nan)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad with NaN to match original length
    result = np.full_like(arr, np.nan)
    start_idx = period - half_period
    end_idx = start_idx + len(hma)
    if end_idx <= len(arr):
        result[start_idx:end_idx] = hma
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d volume confirmation: > 1.5x 20-period average (tight filter to reduce trades) ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Camarilla Pivot Levels (R3, S3) ---
    # Calculate from previous 1d bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r3 = pivot + (range_ * 1.1 / 2.0)  # R3 = pivot + (high-low)*1.1/2
    s3 = pivot - (range_ * 1.1 / 2.0)  # S3 = pivot - (high-low)*1.1/2
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w HMA(21) - trend filter
    hma_21_1w = calculate_hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(hma_21_1w_aligned[i]) or
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1w HMA21 (bullish trend) AND volume confirm
            if (close[i] > r3[i] and 
                close[i] > hma_21_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1w HMA21 (bearish trend) AND volume confirm
            elif (close[i] < s3[i] and 
                  close[i] < hma_21_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 1w HMA21 (mean reversion to trend) OR touches S3 (opposite level)
            if (close[i] <= hma_21_1w_aligned[i] or 
                close[i] < s3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests 1w HMA21 (mean reversion to trend) OR touches R3 (opposite level)
            if (close[i] >= hma_21_1w_aligned[i] or 
                close[i] > r3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals