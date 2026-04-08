#!/usr/bin/env python3
# [24889] 4h_1d1w_camarilla_volume_v1
# Hypothesis: 4-hour Camarilla pivot levels from 1-day and 1-week timeframes with volume confirmation.
# Long when price bounces off L4 level in bullish trend (price > 1-week EMA50).
# Short when price bounces off H4 level in bearish trend (price < 1-week EMA50).
# Uses tight entry conditions (~15-25 trades/year) to minimize fee drag and work in both bull/bear markets.
# Camarilla levels provide precise support/resistance; volume confirms breakout strength; weekly trend filters noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d1w_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day and 1-week data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Calculate 1-day Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 1:  # Need previous day's data
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            rang = ph - pl
            if rang > 0:
                camarilla_h4[i] = pc + (1.1/12) * rang
                camarilla_l4[i] = pc - (1.1/12) * rang
    
    # Calculate 1-week Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_h4_1w = np.full(len(df_1w), np.nan)
    camarilla_l4_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:  # Need previous week's data
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            rang = ph - pl
            if rang > 0:
                camarilla_h4_1w[i] = pc + (1.1/12) * rang
                camarilla_l4_1w[i] = pc - (1.1/12) * rang
    
    # Align Camarilla levels and weekly EMA to 4-hour timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h4_1w = camarilla_h4_1w_aligned[i]
        l4_1w = camarilla_l4_1w_aligned[i]
        weekly_uptrend = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below L4 or volume drops
            if price < l4 or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above H4 or volume drops
            if price > h4 or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price bounces off L4 with volume expansion and weekly uptrend
            if abs(price - l4) / l4 < 0.005 and vol_ratio > 1.6 and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price bounces off H4 with volume expansion and weekly downtrend
            elif abs(price - h4) / h4 < 0.005 and vol_ratio > 1.6 and not weekly_uptrend:
                position = -1
                signals[i] = -0.25
    
    return signals