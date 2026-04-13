#!/usr/bin/env python3
"""
4h_12h_1d_Triple_Pivot_Confluence_Strategy
Hypothesis: Combines 12h Camarilla pivot levels with 1d trend filter and volume confirmation on 4h timeframe to capture high-probability swing trades.
Uses Camarilla H4 and L4 levels from 12h, 4h close above/below 1d EMA50 for trend, and volume > 1.3x 20-period average for confirmation.
Designed to work in both bull and bear markets by trading mean reversion at key pivot levels with trend alignment.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    # 12h Camarilla pivot levels (H4, L4)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        h4_12h = np.full(len(prices), np.nan)
        l4_12h = np.full(len(prices), np.nan)
    else:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Calculate pivot point and ranges
        pivot_12h = (high_12h + low_12h + close_12h) / 3
        range_12h = high_12h - low_12h
        
        # Camarilla levels: H4 = Close + 1.1/2 * Range, L4 = Close - 1.1/2 * Range
        h4_12h_raw = close_12h + (1.1 / 2) * range_12h
        l4_12h_raw = close_12h - (1.1 / 2) * range_12h
        
        # Align to 4h timeframe
        h4_12h = align_htf_to_ltf(prices, df_12h, h4_12h_raw)
        l4_12h = align_htf_to_ltf(prices, df_12h, l4_12h_raw)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        ema50_1d = np.full(len(prices), np.nan)
    else:
        close_1d = df_1d['close'].values
        ema50_1d_raw = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema50_1d = align_htf_to_ltf(prices, df_1d, ema50_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(ema50_1d[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price near L4 support with uptrend and volume
        near_support = low[i] <= l4_12h[i] * 1.002  # within 0.2% of L4
        uptrend = close[i] > ema50_1d[i]
        long_signal = near_support and uptrend and volume_expansion[i]
        
        # Short setup: price near H4 resistance with downtrend and volume
        near_resistance = high[i] >= h4_12h[i] * 0.998  # within 0.2% of H4
        downtrend = close[i] < ema50_1d[i]
        short_signal = near_resistance and downtrend and volume_expansion[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_1d_Triple_Pivot_Confluence_Strategy"
timeframe = "4h"
leverage = 1.0