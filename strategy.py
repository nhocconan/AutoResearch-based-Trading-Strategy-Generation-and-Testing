#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1-week Trend Filter and Volume Spike.
Long when price breaks above 1d Donchian high + 1w trend up + volume spike.
Short when price breaks below 1d Donchian low + 1w trend down + volume spike.
Exit when price returns to opposite Donchian band or trend reverses.
Designed for 7-25 trades/year with strong edge in trending markets.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
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
    
    # Align weekly EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Donchian channels (20-period)
    donch_high = np.empty_like(high, dtype=np.float64)
    donch_low = np.empty_like(low, dtype=np.float64)
    donch_high.fill(np.nan)
    donch_low.fill(np.nan)
    
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (34) + Donchian (20) + Volume MA (20)
    start_idx = max(19, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        trend_1w = ema_1w_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper band + 1w trend up + volume spike
            if price_now > upper_band and price_now > trend_1w and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower band + 1w trend down + volume spike
            elif price_now < lower_band and price_now < trend_1w and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower band or 1w trend turns down
            if price_now < lower_band or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper band or 1w trend turns up
            if price_now > upper_band or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0