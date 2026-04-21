#!/usr/bin/env python3
"""
1d_1w_TradingRangeBreakout
Hypothesis: Trade range breakouts with weekly trend filter on daily chart.
In both bull and bear markets, price consolidates in ranges before breaking out.
We buy when price breaks above weekly resistance with volume confirmation,
sell when breaks below weekly support with volume confirmation.
Weekly trend filter ensures we trade with the higher timeframe trend.
Designed for low trade frequency (target: 10-25/year) to minimize fee drag.
Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend and range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly high and low for range
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly high/low to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day average volume for confirmation
    volume_avg = np.zeros(n)
    for i in range(n):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after volume average warmup
        # Skip if NaN in critical values
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        vol_ok = volume > (1.5 * volume_avg[i])
        
        # Stoploss: 2.5 * ATR (20-period) from entry
        # Calculate ATR on the fly for current position
        if position != 0:
            # Calculate ATR for last 20 periods
            start_idx = max(0, i-20)
            tr = []
            for j in range(start_idx+1, i+1):
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1])
                tr3 = abs(low[j] - close[j-1])
                tr.append(max(tr1, tr2, tr3))
            if len(tr) > 0:
                atr_val = np.mean(tr)
            else:
                atr_val = 0.0
            
            if position == 1 and price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        if position == 0:
            # Long: price breaks above weekly high with volume
            if price > weekly_high_val and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly low with volume
            elif price < weekly_low_val and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls back into weekly range or breaks below weekly low
            if price < weekly_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back into weekly range or breaks above weekly high
            if price > weekly_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_TradingRangeBreakout"
timeframe = "1d"
leverage = 1.0