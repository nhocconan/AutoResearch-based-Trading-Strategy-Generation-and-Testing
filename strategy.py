#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d HMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 and close > 1d HMA34 with volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 and close < 1d HMA34 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Camarilla R3/S3 are stronger intraday levels than R1/S1, reducing false breakouts and overtrading.
# 1d HMA34 ensures higher timeframe trend alignment; volume spike confirms momentum.
# This variant targets fewer, higher-quality trades to avoid fee drag while maintaining edge in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dHMA34_Trend_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate 1d HMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    hma_34_1d = calculate_hma(df_1d['close'].values, 34)
    hma_34_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla R3 = close_prev + (high_prev - low_prev) * 1.1/4
    # Camarilla S3 = close_prev - (high_prev - low_prev) * 1.1/4
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(hma_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d HMA34, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > hma_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d HMA34, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < hma_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3_aligned[i] or 
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