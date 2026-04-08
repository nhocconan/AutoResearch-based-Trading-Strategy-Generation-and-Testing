#!/usr/bin/env python3
# 1d_weekly_pivot_reversion_v1
# Hypothesis: Daily mean reversion from weekly pivot levels with volume confirmation.
# Long when price pulls back to weekly pivot support in an uptrend (price > weekly SMA50).
# Short when price rallies to weekly pivot resistance in a downtrend (price < weekly SMA50).
# Uses weekly pivot from 1-week data as dynamic support/resistance.
# Works in both bull (buy dips) and bear (sell rallies) markets by following trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for weekly pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from weekly data (using previous week's data)
    # Weekly pivot = (Prior week high + prior week low + prior week close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Need at least 1 week
    weekly_pivot = np.full(len(close_1w), np.nan)
    for i in range(0, len(close_1w)):
        if i >= 0:  # Use current week's data for pivot (will be aligned to previous week)
            week_high = high_1w[i]
            week_low = low_1w[i]
            week_close = close_1w[i]
            weekly_pivot[i] = (week_high + week_low + week_close) / 3.0
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align weekly indicators to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate daily volatility filter (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or atr[i] == 0):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        pivot = weekly_pivot_aligned[i]
        sma50 = sma_50_1w_aligned[i]
        vol_ma = np.mean(volume[max(0, i-19):i+1])  # 20-period volume average
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
        
        if position == 1:  # Long
            # Exit: price reaches weekly pivot or trend changes
            if price >= pivot or price < sma50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches weekly pivot or trend changes
            if price <= pivot or price > sma50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price pulls back to weekly pivot support in uptrend
            if price <= pivot * 1.02 and price >= pivot * 0.98 and price > sma50 and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price rallies to weekly pivot resistance in downtrend
            elif price >= pivot * 0.98 and price <= pivot * 1.02 and price < sma50 and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals