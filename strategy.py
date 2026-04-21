#!/usr/bin/env python3
"""
4h_1d_1w_13_34_EMA_Crossover_Volume_Filter
Hypothesis: Combining fast (13) and slow (34) EMA crossovers on 4h timeframe with 1w trend filter and volume confirmation creates a robust trend-following system. The 13/34 crossover captures medium-term momentum shifts, while the weekly EMA34 ensures alignment with higher timeframe trend. Volume confirmation filters out false breakouts. Works in both bull and bear markets by following the trend direction as defined by the weekly EMA. Target 20-40 trades/year to minimize fee drag.
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
    
    # Align weekly EMA34 to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 4h data for EMA13 and EMA34
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 with proper initialization
    ema13 = np.full(n, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[:13])
        alpha13 = 2.0 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha13 * close[i] + (1 - alpha13) * ema13[i-1]
    
    # Calculate EMA34 with proper initialization
    ema34 = np.full(n, np.nan)
    if len(close) >= 34:
        ema34[33] = np.mean(close[:34])
        alpha34 = 2.0 / (34 + 1)
        for i in range(34, n):
            ema34[i] = alpha34 * close[i] + (1 - alpha34) * ema34[i-1]
    
    # Calculate ATR for volatility filter and stoploss
    atr = np.full(n, np.nan)
    if len(close) >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Calculate ATR with Wilder's smoothing
        atr[13] = np.mean(tr[1:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            volume_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(34, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema13[i]) or np.isnan(ema34[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema13_val = ema13[i]
        ema34_val = ema34[i]
        weekly_trend = ema34_1w_aligned[i]
        vol_confirm = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: EMA13 crosses above EMA34 with volume confirmation and price above weekly EMA34 (uptrend)
            if ema13_val > ema34_val and ema13[i-1] <= ema34[i-1] and vol_confirm and price > weekly_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: EMA13 crosses below EMA34 with volume confirmation and price below weekly EMA34 (downtrend)
            elif ema13_val < ema34_val and ema13[i-1] >= ema34[i-1] and vol_confirm and price < weekly_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: EMA13 crosses below EMA34 or price breaks below weekly EMA34
            if ema13_val < ema34_val or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA13 crosses above EMA34 or price breaks above weekly EMA34
            if ema13_val > ema34_val or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_13_34_EMA_Crossover_Volume_Filter"
timeframe = "4h"
leverage = 1.0