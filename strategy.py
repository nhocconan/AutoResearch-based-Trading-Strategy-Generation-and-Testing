#!/usr/bin/env python3
"""
6h_1w_donchian_breakout_v1
Hypothesis: Weekly Donchian breakout with 1d trend filter (EMA 50) and volume confirmation.
Weekly trend provides structural bias; 60-minute entries align with breakout momentum.
Works in bull/bear markets via trend filter + volume/volatility filters reducing whipsaw.
Target: 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Weekly Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = np.full(len(high_1w), np.nan)
    donchian_low_1w = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        donchian_high_1w[i] = np.max(high_1w[i-20:i])
        donchian_low_1w[i] = np.min(low_1w[i-20:i])
    
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.03 * close[i]  # ATR less than 3% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > 1d EMA for longs, price < 1d EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume confirmation, volatility filter, and trend filter
            if close[i] > donchian_high_1w_aligned[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume confirmation, volatility filter, and trend filter
            elif close[i] < donchian_low_1w_aligned[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals