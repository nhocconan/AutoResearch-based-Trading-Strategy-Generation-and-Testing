#!/usr/bin/env python3
"""
12h_1w_Camarilla_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Weekly trend filter + 12h price breaking above R1 or below S1 with volume confirmation captures institutional breakouts. Works in bull/bear markets by aligning with weekly trend. Target 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.zeros_like(close_1w)
    ema34_1w[0] = close_1w[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1w)):
        ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align weekly EMA34 to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (R1, S1) from previous day's OHLC
    R1_1d = np.full(len(df_1d), np.nan)
    S1_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        range_val = phigh - plow
        R1_1d[i] = pclose + (range_val * 1.1 / 12)
        S1_1d[i] = pclose - (range_val * 1.1 / 12)
    
    # Align daily Camarilla levels to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # 12h price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = R1_12h[i]
        s1 = S1_12h[i]
        ema34 = ema34_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Calculate ATR for stoploss (20-period)
        if i >= 20:
            tr_values = []
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
                    tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
        else:
            atr = 0
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in uptrend (price > weekly EMA34)
            if price > r1 and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume confirmation in downtrend (price < weekly EMA34)
            elif price < s1 and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns below weekly EMA34 (trend change)
            if price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly EMA34 (trend change)
            if price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0