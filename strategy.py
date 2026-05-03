#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w HMA21 trend filter and volume confirmation
# Camarilla R3/S3 act as strong intraday support/resistance; breakouts often lead to sustained moves.
# 1w HMA21 ensures trades align with weekly trend to avoid counter-trend whipsaws.
# Volume confirmation (1.5x 20-period EMA) filters low-momentum breakouts.
# Designed for 30-80 total trades over 4 years (7-20/year) with discrete sizing (0.25) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_Camarilla_R3S3_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and 1w data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + 1.1 * camarilla_range / 2
    camarilla_S3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 1d timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Calculate 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA calculation
    wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
    
    # Align 1w HMA21 to 1d timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Volume confirmation: 20-period EMA on 1d
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 1w HMA21
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 1w HMA21 + volume spike
            if close[i] > camarilla_R3_aligned[i] and price_above_hma and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + below 1w HMA21 + volume spike
            elif close[i] < camarilla_S3_aligned[i] and price_below_hma and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses weekly trend alignment
            if close[i] < camarilla_S3_aligned[i] or not price_above_hma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses weekly trend alignment
            if close[i] > camarilla_R3_aligned[i] or not price_below_hma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals