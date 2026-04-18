#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: Donchian channel breakouts on 4h timeframe, filtered by daily trend (EMA34) and volume confirmation, capture institutional moves in both bull and bear markets. Using tight entry conditions (breakout + trend + volume) limits trades to ~20-30/year, reducing fee drag while maintaining edge. Stops via ATR-based trailing stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for stop loss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 34  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_4h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        # Update trailing stop levels
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if price > donchian_high[i] and volume_filter[i] and price > ema_1d_4h[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = high[i]
            # Short: break below Donchian low with volume and downtrend
            elif price < donchian_low[i] and volume_filter[i] and price < ema_1d_4h[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = low[i]
        
        elif position == 1:
            # Check for stop loss: 2 * ATR below highest since entry
            if high[i] < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check for stop loss: 2 * ATR above lowest since entry
            if low[i] > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0