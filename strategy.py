#!/usr/bin/env python3
# [24973] 4h_12h_donchian_volume_trend_v1
# Hypothesis: 4-hour Donchian(20) breakout with 12-hour trend filter (price > EMA50) and volume confirmation.
# Long when price breaks above 20-period high with volume > 2.0x average and price > 12-hour EMA50.
# Short when price breaks below 20-period low with volume > 2.0x average and price < 12-hour EMA50.
# Exit when price returns to 10-period moving average.
# Uses 12-hour EMA50 for trend bias, effective in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12-hour EMA50
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema = np.zeros(len(close_12h))
        ema[0] = close_12h[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_12h)):
            ema[i] = alpha * close_12h[i] + (1 - alpha) * ema[i-1]
        ema50_12h[49:] = ema[49:]
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-period moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12-hour EMA50 to 4-hour timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-period MA
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-period MA
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above EMA50
            if price > donchian_high[i] and vol_ratio > 2.0 and price > ema50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below EMA50
            elif price < donchian_low[i] and vol_ratio > 2.0 and price < ema50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals