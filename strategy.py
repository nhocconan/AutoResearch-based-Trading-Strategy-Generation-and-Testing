#!/usr/bin/env python3
"""
6h_1w1d_volatility_breakout_v1
Hypothesis: 6-hour strategy using weekly volatility breakout with daily trend filter and volume confirmation.
Breakouts occur when price moves beyond weekly ATR-based bands with expanding volume.
Weekly context prevents false breakouts in choppy markets.
Works in both bull (breakouts continue) and bear (breakdowns accelerate) markets.
Target: 20-50 total trades over 4 years (5-12/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w1d_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(len(high))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data for context
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR-based bands (using previous weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    # Weekly volatility bands: center = weekly close, width = 1.5 * ATR
    upper_band_1w = close_1w + (1.5 * atr_1w)
    lower_band_1w = close_1w - (1.5 * atr_1w)
    
    # Calculate daily EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6-hour timeframe
    upper_band_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_band_1w)
    lower_band_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_band_1w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_band_1w_aligned[i]) or np.isnan(lower_band_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper_band = upper_band_1w_aligned[i]
        lower_band = lower_band_1w_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below lower weekly band or volume drops
            if price < lower_band or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above upper weekly band or volume drops
            if price > upper_band or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper weekly band with volume expansion and uptrend
            if price > upper_band and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower weekly band with volume expansion and downtrend
            elif price < lower_band and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals