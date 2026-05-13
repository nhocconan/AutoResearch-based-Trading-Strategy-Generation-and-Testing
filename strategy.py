#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w HMA21 trend filter and volume confirmation.
# Long when price breaks above R3 and close > 1w HMA21 with volume > 1.8x 24-bar average.
# Short when price breaks below S3 and close < 1w HMA21 with volume > 1.8x 24-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# R3/S3 levels act as strong breakout zones; combined with 1w trend filter and volume spike reduces false breakouts.
# Works in bull markets via breakouts and in bear markets via mean-reversion at extreme levels.

name = "12h_Camarilla_R3_S3_Breakout_1wHMA21_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean()
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    lookback = 24  # for volume average (24 * 12h = 12 days)
    
    # Calculate Camarilla levels (R3, S3) using previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each 1d bar: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 = close + (high - low) * 1.1 / 4
    # Camarilla S3 = close - (high - low) * 1.1 / 4
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 1w HMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    hma_21_1w = calculate_hma(df_1w['close'], 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate average volume for confirmation (24-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3, close > 1w HMA21, volume spike
            if (high[i] > R3_aligned[i] and 
                close[i] > hma_21_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3, close < 1w HMA21, volume spike
            elif (low[i] < S3_aligned[i] and 
                  close[i] < hma_21_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR volume drops below average
            if (low[i] < S3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR volume drops below average
            if (high[i] > R3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals