#24770

#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_v1
Hypothesis: 4-hour Donchian breakout with daily trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with volume > 1.5x average and price > daily EMA50 (bullish trend).
Short when price breaks below 20-period Donchian low with volume > 1.5x average and price < daily EMA50 (bearish trend).
Exit when price crosses the Donchian midline (average of 20-period high/low).
Uses discrete position sizing (0.25) to minimize churn. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channel: upper and lower bands"""
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
    if n < 50:
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
    
    # Calculate 4-hour Donchian channel (20-period)
    donch_hi, donch_lo = calculate_donchian(high, low, 20)
    donch_mid = (donch_hi + donch_lo) / 2.0
    
    # Align daily EMA to 4-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donch_hi[i]
        lower = donch_lo[i]
        mid = donch_mid[i]
        trend_up = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian midline
            if price < mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian midline
            if price > mid:
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