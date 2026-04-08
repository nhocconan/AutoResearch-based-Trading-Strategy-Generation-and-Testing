#!/usr/bin/env python3
# [24877] 4h_1d_donchian_volume_breakout_v1
# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1w trend filter.
# Long when price breaks above 1d Donchian high (20) with volume > 2x average and price > 1w EMA200.
# Short when price breaks below 1d Donchian low (20) with volume > 2x average and price < 1w EMA200.
# Exit when price crosses opposite 1d Donchian level OR volume falls below 1.5x average.
# Uses higher timeframe for breakout levels and trend filter to avoid false signals.
# Target: 20-40 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_breakout_v1"
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
    
    # Get 1-day and 1-week data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1w EMA for trend filter (200-period)
    close_1w = df_1w['close'].values
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_200_1w[i-1]
    
    # Align indicators to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        trend_up_1w = price > ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 1d Donchian low or volume drops below 1.5x average
            if price < lower or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 1d Donchian high or volume drops below 1.5x average
            if price > upper or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 1d Donchian high with volume expansion and uptrend on 1w
            if price > upper and vol_ratio > 2.0 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 1d Donchian low with volume expansion and downtrend on 1w
            elif price < lower and vol_ratio > 2.0 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals