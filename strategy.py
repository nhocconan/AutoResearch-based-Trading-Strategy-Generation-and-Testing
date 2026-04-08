#!/usr/bin/env python3
"""
6h_1w_vwap_deviation_mean_reversion_v1
Hypothesis: Price tends to revert to weekly VWAP after significant deviations (>1.5 sigma), 
with mean reversion strongest when weekly trend is sideways (low ADX). Works in bull/bear 
because it fades extremes rather than following trends. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_vwap_deviation_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for VWAP and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for each week
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_values = vwap.values
    
    # Standard deviation of price from VWAP (using typical price)
    price_dev = typical_price - vwap
    std_dev = pd.Series(price_dev).rolling(window=20, min_periods=20).std().values
    
    # ADX for trend filter (low ADX = ranging market, good for mean reversion)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align to 6h
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    std_dev_aligned = align_htf_to_ltf(prices, df_1w, std_dev)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(std_dev_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Skip if no deviation data (std_dev = 0)
        if std_dev_aligned[i] <= 0:
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Deviation from VWAP in sigma units
        deviation = (close[i] - vwap_aligned[i]) / std_dev_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP or ADX increases (trending market)
            if deviation <= 0.2 or adx_aligned[i] > 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or ADX increases (trending market)
            if deviation >= -0.2 or adx_aligned[i] > 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in low ADX (ranging) markets
            if adx_aligned[i] < 20:
                # Long entry: price significantly below VWAP
                if deviation < -1.5:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price significantly above VWAP
                elif deviation > 1.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals