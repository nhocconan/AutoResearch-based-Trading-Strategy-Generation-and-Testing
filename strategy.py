#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Strategy: 12-hour Camarilla pivot breakout with daily trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaks out of Camarilla pivot levels during high volume periods, with daily trend filter to avoid counter-trend trades. Works in bull by capturing breakouts with trend, and in bear by fading false breakouts via trend filter. Volume confirms institutional participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for a given period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Resistance levels
    r4 = close + range_val * 1.1 / 2.0
    r3 = close + range_val * 1.1 / 4.0
    r2 = close + range_val * 1.1 / 6.0
    r1 = close + range_val * 1.1 / 12.0
    
    # Support levels
    s1 = close - range_val * 1.1 / 12.0
    s2 = close - range_val * 1.1 / 6.0
    s3 = close - range_val * 1.1 / 4.0
    s4 = close - range_val * 1.1 / 2.0
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivots for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    r4_1d = np.zeros(len(df_1d))
    r3_1d = np.zeros(len(df_1d))
    r2_1d = np.zeros(len(df_1d))
    r1_1d = np.zeros(len(df_1d))
    pivot_1d = np.zeros(len(df_1d))
    s1_1d = np.zeros(len(df_1d))
    s2_1d = np.zeros(len(df_1d))
    s3_1d = np.zeros(len(df_1d))
    s4_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        r4, r3, r2, r1, p, s1, s2, s3, s4 = calculate_camarilla_pivots(high_1d[i], low_1d[i], close_1d[i])
        r4_1d[i] = r4
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        pivot_1d[i] = p
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align daily Camarilla levels to 12h timeframe (with 1-day delay for confirmation)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d, additional_delay_bars=1)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d, additional_delay_bars=1)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d, additional_delay_bars=1)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d, additional_delay_bars=1)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d, additional_delay_bars=1)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d, additional_delay_bars=1)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d, additional_delay_bars=1)
    
    # Daily EMA for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period) for confirmation on 12h data
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below daily EMA20
        uptrend_1d = price_close > ema_20_1d_aligned[i]
        downtrend_1d = price_close < ema_20_1d_aligned[i]
        
        # Breakout signals: price breaks Camarilla levels with volume
        long_breakout = (price_close > r4_1d_aligned[i]) and vol_spike[i]
        short_breakout = (price_close < s4_1d_aligned[i]) and vol_spike[i]
        
        # Exit when price returns to daily pivot or opposite breakout
        exit_long = position == 1 and (price_close < pivot_1d_aligned[i])
        exit_short = position == -1 and (price_close > pivot_1d_aligned[i])
        
        # Trading logic: only trade in direction of daily trend
        if long_breakout and uptrend_1d and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and downtrend_1d and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals