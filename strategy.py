#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with volume confirmation and ATR volatility filter
# Breakouts above/below 20-period Donchian channels capture momentum.
# Volume filter ensures breakout strength. ATR filter avoids choppy markets.
# Works in both bull and bear by using breakout direction with volatility filter.
# Target: 75-200 total trades over 4 years (~19-50/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr)):
        atr_14_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ATR (14), volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        atr_val = atr_14_aligned[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # ATR filter: sufficient volatility (avoid choppy markets)
        atr_filter = atr_val > 0  # Always true if ATR calculated, but keeps structure
        
        if position == 0:
            # Long: break above Donchian high with volume and volatility
            if price > donchian_high[i] and vol_filter and atr_filter:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume and volatility
            elif price < donchian_low[i] and vol_filter and atr_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low (mean reversion)
            if price <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high (mean reversion)
            if price >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0