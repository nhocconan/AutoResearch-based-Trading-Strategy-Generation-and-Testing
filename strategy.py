#!/usr/bin/env python3
"""
12h_1w_donchian_breakout_v1
Hypothesis: 12-hour strategy using weekly Donchian channel breakout with volume confirmation and trend filter.
Long when price breaks above weekly Donchian high (20) with volume > 1.5x average and price > weekly EMA50 (bullish trend).
Short when price breaks below weekly Donchian low (20) with volume > 1.5x average and price < weekly EMA50 (bearish trend).
Exit when price returns to weekly Donchian midpoint or volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian_channels(high, low, window):
    """Calculate Donchian channels: upper band = rolling max(high), lower band = rolling min(low)"""
    if len(high) < window:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(high, np.nan, dtype=float)
    
    for i in range(window-1, len(high)):
        upper[i] = np.max(high[i-window+1:i+1])
        lower[i] = np.min(low[i-window+1:i+1])
    
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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_high_20, donch_low_20 = calculate_donchian_channels(high_1w, low_1w, 20)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Calculate weekly Donchian midpoint for exit
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Align indicators to 12-hour timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_20_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 30-period average on 12h timeframe
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(donch_mid_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donch_high_20_aligned[i]
        lower = donch_low_20_aligned[i]
        midpoint = donch_mid_20_aligned[i]
        trend_up = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to midpoint or volume drops below average
            if price <= midpoint or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to midpoint or volume drops below average
            if price >= midpoint or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume expansion and uptrend
            if price > upper and vol_ratio > 1.5 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume expansion and downtrend
            elif price < lower and vol_ratio > 1.5 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals