#!/usr/bin/env python3
# 6h_1w_1d_camarilla_confluence_v1
# Strategy: 6s Camarilla pivot confluence with weekly trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla levels (R3/S3, R4/S4) act as strong support/resistance. 
# In uptrend (price > weekly EMA20), buy at S3/S4 with reversal signs. 
# In downtrend (price < weekly EMA20), sell at R3/R4 with rejection signs.
# Weekly EMA filter prevents counter-trend trading in strong trends.
# Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_camarilla_confluence_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), \
               np.full_like(close, np.nan), np.full_like(close, np.nan)
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from weekly EMA20
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Price position relative to Camarilla levels
        near_s3 = abs(close[i] - s3_1d_aligned[i]) / close[i] < 0.002  # Within 0.2%
        near_s4 = abs(close[i] - s4_1d_aligned[i]) / close[i] < 0.002  # Within 0.2%
        near_r3 = abs(close[i] - r3_1d_aligned[i]) / close[i] < 0.002  # Within 0.2%
        near_r4 = abs(close[i] - r4_1d_aligned[i]) / close[i] < 0.002  # Within 0.2%
        
        # Rejection signals: price moves away from level after touching
        rejected_s3 = (near_s3 and i > 20 and 
                      close[i] > close[i-1] and close[i-1] <= s3_1d_aligned[i-1])
        rejected_s4 = (near_s4 and i > 20 and 
                      close[i] > close[i-1] and close[i-1] <= s4_1d_aligned[i-1])
        rejected_r3 = (near_r3 and i > 20 and 
                      close[i] < close[i-1] and close[i-1] >= r3_1d_aligned[i-1])
        rejected_r4 = (near_r4 and i > 20 and 
                      close[i] < close[i-1] and close[i-1] >= r4_1d_aligned[i-1])
        
        # Entry conditions
        # Long: In uptrend, price rejects S3 or S4 support
        if trend_bullish and (rejected_s3 or rejected_s4) and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: In downtrend, price rejects R3 or R4 resistance
        elif trend_bearish and (rejected_r3 or rejected_r4) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite rejection or trend change
        elif position == 1 and (rejected_r3 or rejected_r4 or not trend_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rejected_s3 or rejected_s4 or not trend_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals