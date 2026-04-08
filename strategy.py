#!/usr/bin/env python3
"""
6h_12h1d_breakout_volume_v1
Hypothesis: 6-hour strategy using 12h Donchian breakout + 1d trend filter + volume confirmation.
Long when price breaks above 12h Donchian high (20) with volume > 2x average and price > 1d EMA200.
Short when price breaks below 12h Donchian low (20) with volume > 2x average and price < 1d EMA200.
Exit when price crosses opposite 12h Donchian level OR volume falls below 1.5x average.
Targets 20-50 trades/year per symbol. Works in bull/bear via trend filter and volatility-based breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12-hour and 1-day data for context
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_high_12h = np.full_like(high_12h, np.nan)
    donch_low_12h = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = calculate_ema(df_1d['close'].values, 200)
    
    # Align indicators to 6-hour timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 12h Donchian low or volume drops below 1.5x average
            if price < lower or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 12h Donchian high or volume drops below 1.5x average
            if price > upper or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h Donchian high with volume expansion and uptrend on 1d
            if price > upper and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h Donchian low with volume expansion and downtrend on 1d
            elif price < lower and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals