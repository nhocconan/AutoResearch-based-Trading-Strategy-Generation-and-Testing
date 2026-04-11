#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_reversion_v1
# Strategy: 12-hour Camarilla pivot mean reversion with 1-day trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price reverts to Camarilla pivot levels (H3/L3) during ranging markets, 
# with trend filter to avoid counter-trend trades. Works in both bull and bear by capturing 
# mean reversion within the dominant daily trend. Uses volume confirmation to ensure 
# institutional participation. Targets 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Pivot + 1.1 * Range / 2
    # L3 = Pivot - 1.1 * Range / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + 1.1 * range_1d / 2.0
    l3_1d = pivot_1d - 1.1 * range_1d / 2.0
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 12h timeframe (wait for daily close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Mean reversion signals: touch of Camarilla H3/L3 levels
        # Long when price touches or goes below L3 in uptrend (bounce from support)
        long_signal = (price_close <= l3_1d_aligned[i]) and vol_spike[i] and uptrend_1d
        # Short when price touches or goes above H3 in downtrend (rejection from resistance)
        short_signal = (price_close >= h3_1d_aligned[i]) and vol_spike[i] and downtrend_1d
        
        # Exit when price returns to pivot level
        exit_long = position == 1 and (price_close >= pivot_1d_aligned[i])
        exit_short = position == -1 and (price_close <= pivot_1d_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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