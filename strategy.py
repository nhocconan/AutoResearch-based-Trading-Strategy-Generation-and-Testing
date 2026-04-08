#!/usr/bin/env python3
# 12h_1d_donchian_breakout_v1
# Hypothesis: 12-hour Donchian(20) breakout with 1-day ATR volatility filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.5x average and ATR(14) > 0.5 * ATR(50) (avoid low volatility).
# Short when price breaks below 20-period low with volume > 1.5x average and ATR(14) > 0.5 * ATR(50).
# Exit when price crosses 50-period moving average.
# Uses 1-day ATR for volatility regime filter to avoid choppy markets, effective in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(14) and ATR(50)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr14 = np.full(len(close_1d), np.nan)
    atr50 = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr)):
        atr14[i] = np.mean(tr[i-13:i+1])
    for i in range(50, len(tr)):
        atr50[i] = np.mean(tr[i-49:i+1])
    
    # Align ATRs to 12-hour timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 50-period moving average for exit
    ma_50 = np.full(n, np.nan)
    for i in range(50, n):
        ma_50[i] = np.mean(close[i-50:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_50[i]) or np.isnan(vol_ma[i]) or np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr_ratio = atr14_aligned[i] / atr50_aligned[i] if atr50_aligned[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 50-period MA
            if price < ma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 50-period MA
            if price > ma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and sufficient volatility
            if price > donchian_high[i] and vol_ratio > 1.5 and atr_ratio > 0.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and sufficient volatility
            elif price < donchian_low[i] and vol_ratio > 1.5 and atr_ratio > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals