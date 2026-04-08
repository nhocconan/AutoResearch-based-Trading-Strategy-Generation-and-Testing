#!/usr/bin/env python3
# [24881] 4h_1d1w_donchian_volume_v1
# Hypothesis: 4-hour Donchian breakout with volume confirmation and weekly trend filter.
# Long when price breaks above 20-period Donchian high with volume > 2x average and price > weekly EMA200.
# Short when price breaks below 20-period Donchian low with volume > 2x average and price < weekly EMA200.
# Exit when price crosses opposite Donchian level OR volume falls below 1.5x average.
# Uses weekly trend to filter trades, reducing false signals in counter-trend moves.
# Target: 20-50 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d1w_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_200_1w[i-1]
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Align weekly EMA200 to 4-hour timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        trend_up_1w = price > ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian low or volume drops below 1.5x average
            if price < lower or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian high or volume drops below 1.5x average
            if price > upper or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and weekly uptrend
            if price > upper and vol_ratio > 2.0 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and weekly downtrend
            elif price < lower and vol_ratio > 2.0 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals