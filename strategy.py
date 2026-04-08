#!/usr/bin/env python3
# [24962] 12h_1d_donchian_volume_reversal_v1
# Hypothesis: 12-hour mean reversion strategy using Donchian(20) breakouts with volume confirmation and 1-day trend filter.
# Long when price breaks below Donchian low (oversold) with volume > 2.0x average and price above 1-day EMA50.
# Short when price breaks above Donchian high (overbought) with volume > 2.0x average and price below 1-day EMA50.
# Exit when price returns to Donchian midpoint or volume drops below 1.5x average.
# Designed for ranging markets with mean reversion tendencies, works in both bull and bear regimes by fading extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2/51) + (ema_50[i-1] * (1 - 2/51))
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align EMA50 to 12-hour timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to Donchian midpoint or volume drops below 1.5x average
            if price >= donchian_mid[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to Donchian midpoint or volume drops below 1.5x average
            if price <= donchian_mid[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks below Donchian low (oversold) with volume expansion and above EMA50
            if price < donchian_low[i] and vol_ratio > 2.0 and price > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks above Donchian high (overbought) with volume expansion and below EMA50
            elif price > donchian_high[i] and vol_ratio > 2.0 and price < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals