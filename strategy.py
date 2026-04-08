#!/usr/bin/env python3
"""
6h_1d1w_volatility_breakout_v1
Hypothesis: Breakouts on 6h timeframe with volatility expansion and weekly trend filter.
- Long when price breaks above 6h Donchian(20) high with volatility expansion and weekly uptrend
- Short when price breaks below 6h Donchian(20) low with volatility expansion and weekly downtrend
- Uses 1-week trend filter (price above/below weekly EMA20) to align with higher timeframe momentum
- Volatility filter: current ATR(5) > 1.5 * ATR(20) to ensure breakouts occur during expansion
- Designed for low trade frequency (15-35/year) to minimize fee drag
- Works in bull/bear via volatility expansion and weekly trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full(len(high), np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(high), np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

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
    
    # Get daily data for weekly aggregation (we'll use 1d to derive weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate 6h ATR for volatility filter
    atr_5 = calculate_atr(high, low, close, 5)
    atr_20 = calculate_atr(high, low, close, 20)
    
    # Calculate weekly EMA from daily data (approximate weekly using 5-day EMA)
    close_1d = df_1d['close'].values
    ema_5_1d = calculate_ema(close_1d, 5)  # 5-day EMA approximates weekly
    
    # Align indicators to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    atr_5_aligned = align_htf_to_ltf(prices, df_1d, atr_5)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    ema_5_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_5_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_5_aligned[i]) or np.isnan(atr_20_aligned[i]) or
            np.isnan(ema_5_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        atr5 = atr_5_aligned[i]
        atr20 = atr_20_aligned[i]
        weekly_ema = ema_5_1d_aligned[i]
        
        # Volatility expansion condition
        vol_expansion = atr5 > 1.5 * atr20
        
        if position == 1:  # Long
            # Exit: price closes below Donchian lower or volatility contraction
            if price < lower or not vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian upper or volatility contraction
            if price > upper or not vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper with volatility expansion and weekly uptrend
            if price > upper and vol_expansion and price > weekly_ema:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower with volatility expansion and weekly downtrend
            elif price < lower and vol_expansion and price < weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals