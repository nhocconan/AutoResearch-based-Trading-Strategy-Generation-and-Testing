#!/usr/bin/env python3
"""
1h_4d_Donchian_Breakout_TrendFilter
Hypothesis: In trending markets, price breaks of the 4-hour Donchian channel (20-period) 
continuation signals align with the daily trend. Use the daily EMA50 as trend filter: 
only take long breaks when price > daily EMA50, short breaks when price < daily EMA50. 
This avoids counter-trend breakouts that fail in ranging/choppy markets. 
Entry occurs on the 1-hour break of the prior 4-hour Donchian high/low. 
Position size: 0.20 to limit drawdown. Expect 15-25 trades/year (~60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i < 20:
            donchian_high[i] = np.max(high_4h[max(0, i-19):i+1]) if i >= 0 else high_4h[i]
            donchian_low[i] = np.min(low_4h[max(0, i-19):i+1]) if i >= 0 else low_4h[i]
        else:
            donchian_high[i] = np.max(high_4h[i-20:i])
            donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h timeframe (waits for 4h bar close)
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or np.isnan(ema50_1d_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        dh = donchian_high_1h[i]  # Prior 4h Donchian high
        dl = donchian_low_1h[i]   # Prior 4h Donchian low
        ema50 = ema50_1d_1h[i]    # Daily EMA50 trend filter
        
        # Stoploss: 2.5 * ATR(14) from entry
        if i >= 14:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            # Calculate ATR(14) using simple mean of last 14 true ranges
            tr_sum = 0
            for j in range(1, 15):
                idx = i - j
                if idx >= 0:
                    tr1_j = high[idx] - low[idx]
                    tr2_j = abs(high[idx] - close[idx-1])
                    tr3_j = abs(low[idx] - close[idx-1])
                    tr_sum += max(tr1_j, tr2_j, tr3_j)
            atr = tr_sum / 14
        else:
            atr = 0
        
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above prior 4h Donchian high AND price > daily EMA50 (uptrend)
            if price > dh and price > ema50:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below prior 4h Donchian low AND price < daily EMA50 (downtrend)
            elif price < dl and price < ema50:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below prior 4h Donchian low OR trend fails
            if price < dl or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above prior 4h Donchian high OR trend fails
            if price > dh or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4d_Donchian_Breakout_TrendFilter"
timeframe = "1h"
leverage = 1.0