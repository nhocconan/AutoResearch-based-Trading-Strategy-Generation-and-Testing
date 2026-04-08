#!/usr/bin/env python3
"""
4h_1d_donchian_volume_crossover_v1
Hypothesis: Use daily Donchian channel (20) breakout with volume confirmation on 4h timeframe.
- Long when price breaks above 4h Donchian high (20) with volume > 1.5x 20-period average
- Short when price breaks below 4h Donchian low (20) with volume > 1.5x 20-period average
- Use daily trend filter (price above/below daily EMA50) to avoid counter-trend trades
- Designed for 4h timeframe with ~20-50 trades/year to minimize fee drag
- Works in bull/bear via trend filter and volatility-based entry conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_crossover_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    if len(high) < period:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Calculate 4h Donchian channel (20 periods)
    donchian_high, donchian_low = calculate_donchian(high, low, 20)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        trend_up = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below Donchian low or volume drops significantly
            if price < lower or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian high or volume drops significantly
            if price > upper or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and uptrend
            if price > upper and vol_ratio > 1.5 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and downtrend
            elif price < lower and vol_ratio > 1.5 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals