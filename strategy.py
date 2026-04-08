#!/usr/bin/env python3
# 4h_1d_atr_breakout_v1
# Hypothesis: 4-hour ATR-based breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above high + ATR multiplier, volume > 1.5x average, and price above 1-day EMA50.
# Short when price breaks below low - ATR multiplier, volume > 1.5x average, and price below 1-day EMA50.
# Exit when price returns to 4-period EMA.
# Uses 1-day EMA50 for trend bias to avoid counter-trend trades.
# Designed to generate ~20-40 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_atr_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Calculate ATR (14-period)
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr[13] = np.mean(tr[1:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 4-period EMA for exit
    ema4 = np.full(n, np.nan)
    if n >= 4:
        ema4[3] = np.mean(close[:4])
        alpha = 2.0 / (4 + 1)
        for i in range(4, n):
            ema4[i] = alpha * close[i] + (1 - alpha) * ema4[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(ema4[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        ema4_val = ema4[i]
        ema50_val = ema50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to 4-period EMA
            if price <= ema4_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 4-period EMA
            if price >= ema4_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: volatility breakout with volume and trend filter
            # Calculate 20-period volume average
            vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else 0
            vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
            
            # Enter long: price breaks above high + 0.5*ATR, volume expansion, above 1d EMA50
            if price > high[i-1] + 0.5 * atr_val and vol_ratio > 1.5 and price > ema50_val:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below low - 0.5*ATR, volume expansion, below 1d EMA50
            elif price < low[i-1] - 0.5 * atr_val and vol_ratio > 1.5 and price < ema50_val:
                position = -1
                signals[i] = -0.25
    
    return signals