#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 12h_Camarilla_Pivot_V4
# Hypothesis: 12-hour strategy using 1-day Camarilla pivot levels with volume confirmation and 1-week trend filter.
# Long when price breaks above 1d R2 with volume > 1.8x average and price > 1w EMA50.
# Short when price breaks below 1d S2 with volume > 1.8x average and price < 1w EMA50.
# Exit when price crosses opposite 1d level OR volume falls below 1.2x average.
# Uses higher timeframe pivots for better signal quality in both bull and bear markets.
# Target: 15-35 trades/year per symbol.

name = "12h_Camarilla_Pivot_V4"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day and 1-week data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Pivot (using previous 1d bar's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # 1d support/resistance levels (Camarilla S2/R2)
    S2_1d = pivot_1d - (range_1d * 1.1 / 6)  # 1d S2
    R2_1d = pivot_1d + (range_1d * 1.1 / 6)  # 1d R2
    
    # Calculate 1w EMA for trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    
    # Align indicators to 12-hour timeframe
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S2_1d_aligned[i]) or np.isnan(R2_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S2 = S2_1d_aligned[i]
        R2 = R2_1d_aligned[i]
        trend_up_1w = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 1d S2 or volume drops below 1.2x average
            if price < S2 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 1d R2 or volume drops below 1.2x average
            if price > R2 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 1d R2 with volume expansion and uptrend on 1w
            if price > R2 and vol_ratio > 1.8 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 1d S2 with volume expansion and downtrend on 1w
            elif price < S2 and vol_ratio > 1.8 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals