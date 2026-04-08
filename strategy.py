#!/usr/bin/env python3
# [24882] 12h_1d1w_donchian_breakout_v1
# Hypothesis: 12-hour strategy using Donchian(20) breakout with volume confirmation and 1-day trend filter.
# Long when price breaks above Donchian upper band (20-period high) on 12h with volume > 2x average and price > 1d EMA50.
# Short when price breaks below Donchian lower band (20-period low) on 12h with volume > 2x average and price < 1d EMA50.
# Exit when price crosses opposite Donchian band OR volume falls below 1.5x average.
# Uses higher timeframe breakouts for better signal quality in both bull and bear markets.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d1w_donchian_breakout_v1"
timeframe = "12h"
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
    if n < 50:
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
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    donchian_upper = np.full_like(high_12h, np.nan)
    donchian_lower = np.full_like(low_12h, np.nan)
    
    for i in range(19, len(high_12h)):  # 20-period lookback
        donchian_upper[i] = np.max(high_12h[i-19:i+1])
        donchian_lower[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 1d EMA for trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    
    # Align indicators to 12-hour timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian lower band or volume drops below 1.5x average
            if price < lower or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian upper band or volume drops below 1.5x average
            if price > upper or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper band with volume expansion and uptrend on 1d
            if price > upper and vol_ratio > 2.0 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band with volume expansion and downtrend on 1d
            elif price < lower and vol_ratio > 2.0 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals