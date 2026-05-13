#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d HMA21 trend filter and volume spike confirmation.
# Long when price breaks above R3 and close > 1d HMA21 with volume > 2.0x 20-bar average.
# Short when price breaks below S3 and close < 1d HMA21 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Camarilla levels provide adaptive support/resistance; 1d HMA21 ensures trend alignment; volume spike confirms momentum.
# Works in bull markets via breakouts and in bear markets via mean-reversion at extreme levels.

name = "12h_Camarilla_R3_S3_Breakout_1dHMA21_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    lookback = 20  # for volume average and Camarilla calculation
    
    # Calculate 1d HMA21 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Camarilla levels (based on previous period)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous lookback period
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3, close > 1d HMA21, volume spike
            if (high[i] > R3[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3, close < 1d HMA21, volume spike
            elif (low[i] < S3[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR volume drops below average
            if (low[i] < S3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR volume drops below average
            if (high[i] > R3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals