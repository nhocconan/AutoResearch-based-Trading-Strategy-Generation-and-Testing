#!/usr/bin/env python3
# [24988] 12h_1w_1d_breakout_volume_volatility_filter_v1
# Hypothesis: 12-hour price breakouts with volume expansion and volatility filter.
# Long when price breaks above 12-period high + volume > 1.5x average + volatility regime (ATR ratio > 0.8)
# Short when price breaks below 12-period low + volume > 1.5x average + volatility regime (ATR ratio > 0.8)
# Exit when price returns to 6-period moving average.
# Uses 1-week trend filter: only trade in direction of weekly EMA(20) to avoid counter-trend whipsaws.
# Designed for low frequency (~20-40 trades/year) to minimize fee drag while capturing momentum bursts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_breakout_volume_volatility_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema_20_1w[i] = np.mean(close_1w[:20])
        else:
            ema_20_1w[i] = (close_1w[i] * 2/21) + (ema_20_1w[i-1] * 19/21)
    
    # Align weekly EMA to 12-hour timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 12-period high/low for breakout
    high_12 = np.full(n, np.nan)
    low_12 = np.full(n, np.nan)
    for i in range(12, n):
        high_12[i] = np.max(high[i-12:i])
        low_12[i] = np.min(low[i-12:i])
    
    # Calculate 6-period moving average for exit
    ma_6 = np.full(n, np.nan)
    for i in range(6, n):
        ma_6[i] = np.mean(close[i-6:i])
    
    # Calculate volume moving average (12-period)
    vol_ma = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma[i] = np.mean(volume[i-12:i])
    
    # Calculate ATR(12) for volatility filter
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(12, n):
        atr[i] = np.mean(tr[i-12:i])
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma = np.full(n, np.nan)
    for i in range(50, n):
        atr_ma[i] = np.mean(atr[i-50:i])
    
    atr_ratio = np.full(n, np.nan)
    for i in range(50, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(high_12[i]) or np.isnan(low_12[i]) or 
            np.isnan(ma_6[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 6-period MA
            if price <= ma_6[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 6-period MA
            if price >= ma_6[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12-period high with volume expansion and ATR ratio > 0.8 and above weekly EMA
            if price > high_12[i] and vol_ratio > 1.5 and atr_ratio[i] > 0.8 and price > ema_20_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12-period low with volume expansion and ATR ratio > 0.8 and below weekly EMA
            elif price < low_12[i] and vol_ratio > 1.5 and atr_ratio[i] > 0.8 and price < ema_20_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals