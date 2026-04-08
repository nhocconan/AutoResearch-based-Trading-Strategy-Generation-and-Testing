#!/usr/bin/env python3
# 4h_donchian20_volatility_breakout_v3
# Hypothesis: Donchian channel breakouts with volatility filter on 4h timeframe.
# Long when price breaks above 20-period high with ATR(14) > 0.5 * ATR(50).
# Short when price breaks below 20-period low with ATR(14) > 0.5 * ATR(50).
# Exit when price crosses the opposite Donchian level.
# Uses 1d Donchian for trend filter: long only if price > 1d Donchian mid, short only if price < 1d Donchian mid.
# Target: 20-40 trades/year with strict volatility and trend filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volatility_breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h and 1d data (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20-1, n):
        donchian_high[i] = np.max(high[i-20+1:i+1])
        donchian_low[i] = np.min(low[i-20+1:i+1])
    
    # 1d Donchian mid for trend filter
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    donchian_1d_high = np.full(len(prev_1d_high), np.nan)
    donchian_1d_low = np.full(len(prev_1d_low), np.nan)
    for i in range(20-1, len(prev_1d_high)):
        donchian_1d_high[i] = np.max(prev_1d_high[i-20+1:i+1])
        donchian_1d_low[i] = np.min(prev_1d_low[i-20+1:i+1])
    donchian_1d_mid = (donchian_1d_high + donchian_1d_low) / 2
    donchian_1d_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_mid)
    
    # ATR for volatility filter
    atr_period = 14
    atr_slow_period = 50
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    atr_slow = np.full(n, np.nan)
    for i in range(atr_slow_period, n):
        atr_slow[i] = np.mean(tr[i-atr_slow_period+1:i+1])
    
    volatility_filter = (atr > 0.5 * atr_slow) & ~np.isnan(atr_slow)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, atr_slow_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_1d_mid_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below 4h Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 4h Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above 4h Donchian high + volatility filter + price > 1d Donchian mid
            if (close[i] > donchian_high[i] and 
                volatility_filter[i] and 
                close[i] > donchian_1d_mid_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 4h Donchian low + volatility filter + price < 1d Donchian mid
            elif (close[i] < donchian_low[i] and 
                  volatility_filter[i] and 
                  close[i] < donchian_1d_mid_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals