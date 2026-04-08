#!/usr/bin/env python3
# 4h_12h_donchian_breakout_v1
# Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high, with price above 12h EMA50 and volume > 1.5x average.
# Short when price breaks below 20-period Donchian low, with price below 12h EMA50 and volume > 1.5x average.
# Exit when price returns to the Donchian midpoint or reverses.
# Designed to generate ~20-40 trades/year to minimize fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(period - 1, n):
        donchian_high[i] = np.max(high[i - period + 1:i + 1])
        donchian_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Donchian midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= period:
            vol_sum -= volume[i - period]
        if i >= period - 1:
            vol_ma[i] = vol_sum / period
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    multiplier = 2 / (ema_period + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h[i] = close_12h[i]
        elif not np.isnan(close_12h[i]):
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
        else:
            ema_12h[i] = ema_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 1:  # Long
            # Exit: price returns to midpoint or reverses below EMA
            if price <= donchian_mid[i] or price < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to midpoint or reverses above EMA
            if price >= donchian_mid[i] or price > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with trend and volume filter
            # Bullish: price breaks above Donchian high, above 12h EMA50, and high volume
            if price > donchian_high[i] and price > ema_12h_aligned[i] and vol > 1.5 * vol_ma[i]:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below Donchian low, below 12h EMA50, and high volume
            elif price < donchian_low[i] and price < ema_12h_aligned[i] and vol > 1.5 * vol_ma[i]:
                position = -1
                signals[i] = -0.25
    
    return signals