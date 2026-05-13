#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w HMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 and close > 1w HMA50 with volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S3 and close < 1w HMA50 with volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# 1w HMA50 ensures higher timeframe trend alignment; volume spike confirms momentum.
# This variant targets fewer, higher-quality trades to avoid fee drag while maintaining edge in both bull and bear markets.

name = "1d_Camarilla_R3_S3_Breakout_1wHMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w HMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    hma_50_1w = calculate_hma(df_1w['close'].values, 50)
    hma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_50_1w)
    
    # Calculate Camarilla levels from previous day
    # Camarilla R3 = close_prev + (high_prev - low_prev) * 1.1/4
    # Camarilla S3 = close_prev - (high_prev - low_prev) * 1.1/4
    close_prev = prices['close'].shift(1).values
    high_prev = prices['high'].shift(1).values
    low_prev = prices['low'].shift(1).values
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(hma_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1w HMA50, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > hma_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1w HMA50, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < hma_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean()
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values