#!/usr/bin/env python3
"""
12h_TripleConfirmation_Strategy
Breakout strategy combining 12h Donchian breakout with 1d volume confirmation and 1d trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average and price > 1d EMA50.
Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average and price < 1d EMA50.
Exit when price returns to Donchian midpoint or trend filter fails.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mft_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donch_len = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(donch_len - 1, n):
        donch_high[i] = np.max(high[i - donch_len + 1:i + 1])
        donch_low[i] = np.min(low[i - donch_len + 1:i + 1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(donch_len - 1, n):
        vol_avg[i] = np.mean(volume[i - donch_len + 1:i + 1])
    
    # Get 1d data for higher timeframe filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume avg, and EMA1d
    start_idx = max(donch_len - 1, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        vol_avg_val = vol_avg[i]
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume confirmation + uptrend
            if (price > donch_high_val and vol > 1.5 * vol_avg_val and price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: Donchian breakout down + volume confirmation + downtrend
            elif (price < donch_low_val and vol > 1.5 * vol_avg_val and price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or trend fails
            if price < donch_mid_val or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint or trend fails
            if price > donch_mid_val or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_TripleConfirmation_Strategy"
timeframe = "12h"
leverage = 1.0