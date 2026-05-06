#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and session filter (08-20 UTC)
# Uses Camarilla pivot levels for intraday structure, 4h EMA50 for trend alignment to avoid counter-trend trades
# Session filter reduces noise during low-liquidity hours. Discrete sizing 0.20 to limit fee drag.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_4hEMA50_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoids datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rang = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rang
    camarilla_s3 = close_1d - 1.1 * rang
    
    # Align HTF indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Camarilla R3 AND uptrend (price > 4h EMA50)
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < Camarilla S3 AND downtrend (price < 4h EMA50)
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests Camarilla S3 from above (trend reversal)
            if close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests Camarilla R3 from below (trend reversal)
            if close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals