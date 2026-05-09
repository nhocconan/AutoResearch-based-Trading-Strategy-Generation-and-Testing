# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h 1w/1d CCI Divergence with Weekly Trend Filter
Long when: Price makes lower low, CCI makes higher low (bullish divergence) above -100, weekly CCI > 0
Short when: Price makes higher high, CCI makes lower high (bearish divergence) below 100, weekly CCI < 0
Exit: CCI crosses zero in opposite direction
Uses weekly CCI for trend filter and daily CCI for divergence signals to avoid false breakouts in ranging markets.
Designed for 6h timeframe with ~50-150 trades over 4 years (12-37/year).
"""

name = "6h_CCI_Divergence_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly CCI (20-period)
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    ma_1w = typical_price_1w.rolling(window=20, min_periods=20).mean()
    mad_1w = typical_price_1w.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_1w = (typical_price_1w - ma_1w) / (0.015 * mad_1w)
    cci_1w = cci_1w.fillna(0).values
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Get daily data for signal generation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily CCI (14-period)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    ma_1d = typical_price_1d.rolling(window=14, min_periods=14).mean()
    mad_1d = typical_price_1d.rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_1d = (typical_price_1d - ma_1d) / (0.015 * mad_1d)
    cci_1d = cci_1d.fillna(0).values
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Align price arrays for divergence detection
    high_aligned = align_htf_to_ltf(prices, df_1d, high)
    low_aligned = align_htf_to_ltf(prices, df_1d, low)
    close_aligned = align_htf_to_ltf(prices, df_1d, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for CCI calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cci_1w_aligned[i]) or np.isnan(cci_1d_aligned[i]) or
            np.isnan(high_aligned[i]) or np.isnan(low_aligned[i]) or np.isnan(close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values for divergence detection (look back 2-5 periods)
        lookback = 3
        if i - lookback < 0:
            continue
            
        # Current and past values
        cci_now = cci_1d_aligned[i]
        cci_past = cci_1d_aligned[i - lookback]
        price_high_now = high_aligned[i]
        price_high_past = high_aligned[i - lookback]
        price_low_now = low_aligned[i]
        price_low_past = low_aligned[i - lookback]
        
        if position == 0:
            # Bullish divergence: price makes lower low, CCI makes higher low
            bull_div = (price_low_now < price_low_past) and (cci_now > cci_past)
            # Bearish divergence: price makes higher high, CCI makes lower high
            bear_div = (price_high_now > price_high_past) and (cci_now < cci_past)
            
            # Enter long: bullish divergence above -100, weekly CCI positive
            if bull_div and (cci_now > -100) and (cci_1w_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish divergence below 100, weekly CCI negative
            elif bear_div and (cci_now < 100) and (cci_1w_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CCI crosses below zero
            if cci_now < 0 and cci_past >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI crosses above zero
            if cci_now > 0 and cci_past <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals