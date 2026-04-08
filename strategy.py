#!/usr/bin/env python3
"""
4h_12h_donchian_volume_v1
Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
- Long when price breaks above Donchian(20) high with volume expansion and 12h uptrend
- Short when price breaks below Donchian(20) low with volume expansion and 12h downtrend
- Uses ATR-based stop loss to limit drawdown
- Designed for moderate trade frequency (20-40/year) to balance opportunity and cost
- Works in bull/bear via 12h trend filter avoiding counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    if len(high) < period:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(high), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (tr[i] + (period-1) * atr[i-1]) / period
    
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    donchian_high, donchian_low = calculate_donchian(high, low, 20)
    
    # Calculate 4h ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = calculate_ema(close_12h, 50)
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, high, donchian_high)  # Use high as dummy for alignment
    donchian_low_aligned = align_htf_to_ltf(prices, high, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, high, atr)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_val = atr_aligned[i]
        trend_up = price > ema_50_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below Donchian low or ATR-based stop
            if price < lower or price < (entry_price - 2.0 * atr_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian high or ATR-based stop
            if price > upper or price > (entry_price + 2.0 * atr_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and uptrend
            if price > upper and vol_ratio > 1.5 and trend_up:
                position = 1
                entry_price = price
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and downtrend
            elif price < lower and vol_ratio > 1.5 and not trend_up:
                position = -1
                entry_price = price
                signals[i] = -0.25
    
    return signals