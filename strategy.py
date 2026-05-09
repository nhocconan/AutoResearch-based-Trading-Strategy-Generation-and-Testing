#!/usr/bin/env python3
# 1d_VWAP_Reversal_1wTrend
# Hypothesis: Daily VWAP mean-reversion with weekly trend filter. In weekly uptrend (price > weekly VWAP), go long when price touches daily VWAP from below; in weekly downtrend (price < weekly VWAP), go short when price touches daily VWAP from above. Uses volume-weighted price for institutional-level support/resistance. Designed for 10-25 trades/year on 1d timeframe.

name = "1d_VWAP_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP (volume-weighted average price)
    vwap_1w = np.full_like(close_1w, np.nan)
    cumulative_volume = 0.0
    cumulative_price_volume = 0.0
    
    for i in range(len(close_1w)):
        typical_price = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        price_volume = typical_price * volume_1w[i]
        cumulative_volume += volume_1w[i]
        cumulative_price_volume += price_volume
        if cumulative_volume > 0:
            vwap_1w[i] = cumulative_price_volume / cumulative_volume
    
    # Align weekly VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (resets daily)
    vwap_1d = np.full_like(close_1d, np.nan)
    cumulative_volume = 0.0
    cumulative_price_volume = 0.0
    
    for i in range(len(close_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        price_volume = typical_price * volume_1d[i]
        cumulative_volume += volume_1d[i]
        cumulative_price_volume += price_volume
        if cumulative_volume > 0:
            vwap_1d[i] = cumulative_price_volume / cumulative_volume
    
    # Align daily VWAP to daily timeframe (no alignment needed as it's same timeframe)
    vwap_1d_aligned = vwap_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(vwap_1w_aligned[i]) or np.isnan(vwap_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price touches daily VWAP from below AND weekly uptrend (price > weekly VWAP)
            if close[i] <= vwap_1d_aligned[i] and close[i] > vwap_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price touches daily VWAP from above AND weekly downtrend (price < weekly VWAP)
            elif close[i] >= vwap_1d_aligned[i] and close[i] < vwap_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above daily VWAP or weekly trend turns down
            if close[i] > vwap_1d_aligned[i] or close[i] < vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses below daily VWAP or weekly trend turns up
            if close[i] < vwap_1d_aligned[i] or close[i] > vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals