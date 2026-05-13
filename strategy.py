#!/usr/bin/env python3
"""
4h_12h_Camarilla_R3_S3_Breakout_Trend
Hypothesis: Camarilla pivot levels (R3/S3) from 12h act as support/resistance.
Breakouts with volume confirmation and aligned 12h trend capture trends in both bull and bear markets.
Designed for low trade frequency (~25-35/year) to minimize fee drag and work across market regimes.
"""

name = "4h_12h_Camarilla_R3_S3_Breakout_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: R3, R4, S3, S4"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = close + range_val * 1.1 / 2
    s3 = close - range_val * 1.1 / 2
    return r3, s3

def calculate_ema(arr, period):
    """Calculate EMA with proper min_periods"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    sma = np.full_like(arr, np.nan, dtype=float)
    sma[period-1] = np.mean(arr[:period])
    multiplier = 2 / (period + 1)
    ema = np.full_like(arr, np.nan, dtype=float)
    ema[period-1] = sma[period-1]
    for i in range(period, len(arr)):
        ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla R3/S3 levels on 12h
    r3_12h, s3_12h = calculate_camarilla(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values
    )
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = calculate_ema(df_12h['close'].values, 50)
    
    # Align 12h indicators to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3_12h)
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Break above R3 with volume confirmation and uptrend (price > EMA50)
            if (close[i] > r3_4h[i] and 
                close[i-1] <= r3_4h[i-1] and 
                volume_confirm[i] and 
                close[i] > ema50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume confirmation and downtrend (price < EMA50)
            elif (close[i] < s3_4h[i] and 
                  close[i-1] >= s3_4h[i-1] and 
                  volume_confirm[i] and 
                  close[i] < ema50_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 (failed breakout) or reaches opposite S3
            if close[i] < r3_4h[i] or close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 (failed breakdown) or reaches opposite R3
            if close[i] > s3_4h[i] or close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals