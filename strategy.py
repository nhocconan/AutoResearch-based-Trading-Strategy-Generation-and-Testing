#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 20-day high + 1w trend up + volume spike.
Short when price breaks below 20-day low + 1w trend down + volume spike.
Exit when price returns to 10-day moving average or trend reverses.
Designed to generate 10-25 trades/year per symbol with strong edge in bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.empty_like(close_1w, dtype=np.float64)
    ema_1w.fill(np.nan)
    alpha = 2.0 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w[i] = close_1w[i]
        elif np.isnan(ema_1w[i-1]):
            ema_1w[i] = close_1w[i]
        else:
            ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align 1w EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 20-day Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 10-day moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(9, n):
        ma_10[i] = np.mean(close[i-9:i+1])
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (34), MA (10), volume MA (20)
    start_idx = max(19, 34, 9, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(ma_10[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        trend_1w = ema_1w_aligned[i]
        ma_level = ma_10[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper channel + 1w trend up + volume spike
            if price_now > upper_channel and price_now > trend_1w and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower channel + 1w trend down + volume spike
            elif price_now < lower_channel and price_now < trend_1w and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 10-day MA or 1w trend turns down
            if price_now < ma_10[i] or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to 10-day MA or 1w trend turns up
            if price_now > ma_10[i] or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0