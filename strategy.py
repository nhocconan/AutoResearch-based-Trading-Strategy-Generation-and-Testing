#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d HMA21 trend filter and volume confirmation
# Long when price breaks above R3 AND 1d HMA21 rising (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below S3 AND 1d HMA21 falling (downtrend) AND volume > 1.5x 20 EMA
# Uses 4h for primary signals (balanced trade frequency), 1d for trend to avoid counter-trend trades.
# Discrete sizing (0.25) to balance return and fee drag. Target: 20-50 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Camarilla_R3S3_1dHMA21_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half/weights_half.sum(), mode='valid')
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full/weights_full.sum(), mode='valid')
    
    # HMA calculation
    raw_hma = 2 * wma_half - wma_full
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt/weights_sqrt.sum(), mode='valid')
    
    # Pad with NaN to match original length
    hma_full = np.full_like(arr, np.nan)
    start_idx = period - len(hma)
    hma_full[start_idx:] = hma
    return hma_full

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d HMA21 for trend filter
    hma_21_1d = calculate_hma(close_1d, 21)
    # Uptrend when HMA rising, downtrend when HMA falling
    hma_rising = np.diff(hma_21_1d, prepend=np.nan) > 0
    hma_falling = np.diff(hma_21_1d, prepend=np.nan) < 0
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1d HMA rising AND volume spike
            if (close[i] > r3_aligned[i] and 
                hma_rising_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND 1d HMA falling AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  hma_falling_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 1d HMA starts falling
            if (close[i] < s3_aligned[i] or 
                hma_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR 1d HMA starts rising
            if (close[i] > r3_aligned[i] or 
                hma_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals