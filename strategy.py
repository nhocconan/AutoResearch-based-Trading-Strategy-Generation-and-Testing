#!/usr/bin/env python3
# [24897] 4h_1d_camarilla_pivot_v1
# Hypothesis: 4-hour Camarilla pivot reversal with volume confirmation and 1-day trend filter.
# Long when price touches or crosses above Camarilla L3 level with volume > 1.3x average and price > 1-day EMA200.
# Short when price touches or crosses below Camarilla H3 level with volume > 1.3x average and price < 1-day EMA200.
# Exit when price reaches opposite H3/L3 level or reverses at L4/H4 levels.
# Uses Camarilla pivots from daily chart for institutional support/resistance, volume filter to avoid false breakouts,
# and EMA200 trend filter to align with higher timeframe direction. Designed for low-frequency, high-quality signals
# (~20-30 trades/year) to minimize fee drag and work in both bull and bear markets by trading mean reversion
# at key institutional levels during ranging markets and trend continuation in trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        if np.isnan(ph) or np.isnan(pl) or np.isnan(pc):
            continue
            
        rang = ph - pl
        camarilla_h4[i] = pc + 1.5 * rang
        camarilla_h3[i] = pc + 1.25 * rang
        camarilla_l3[i] = pc - 1.25 * rang
        camarilla_l4[i] = pc - 1.5 * rang
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day indicators to 4-hour timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price reaches H3 level or reverses at H4
            if price >= camarilla_h3_aligned[i] or price >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches L3 level or reverses at L4
            if price <= camarilla_l3_aligned[i] or price <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches or crosses above L3 with volume expansion and uptrend bias
            if price >= camarilla_l3_aligned[i] and vol_ratio > 1.3 and price > ema_200_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches or crosses below H3 with volume expansion and downtrend bias
            elif price <= camarilla_h3_aligned[i] and vol_ratio > 1.3 and price < ema_200_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals