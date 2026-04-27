#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + 1w Trend Filter
Long when price breaks above Donchian(20) high + volume spike + 1w trend up
Short when price breaks below Donchian(20) low + volume spike + 1w trend down
Exit when price crosses Donchian midpoint or trend reverses
Target: 20-40 trades/year per symbol, focus on breakouts with institutional volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2.0
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.empty_like(close_1w, dtype=np.float64)
    ema_1w.fill(np.nan)
    if len(close_1w) >= 34:
        alpha = 2.0 / (34 + 1)
        for i in range(len(close_1w)):
            if i == 0:
                ema_1w[i] = close_1w[i]
            elif np.isnan(ema_1w[i-1]):
                ema_1w[i] = close_1w[i]
            else:
                ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to 4h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 2.5x average (institutional participation)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20) + volume MA(20) + weekly EMA(34)
    start_idx = max(19, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper = donch_high[i]
        lower = donch_low[i]
        midpoint = donch_mid[i]
        trend_1w = ema_1w_aligned[i]
        
        # Volume filter: volume > 2.5x average
        vol_filter = vol_now > 2.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above upper band + uptrend + volume spike
            if price_now > upper and price_now > trend_1w and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + downtrend + volume spike
            elif price_now < lower and price_now < trend_1w and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses midpoint OR trend turns down
            if price_now < midpoint or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses midpoint OR trend turns up
            if price_now > midpoint or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_1wTrend"
timeframe = "4h"
leverage = 1.0