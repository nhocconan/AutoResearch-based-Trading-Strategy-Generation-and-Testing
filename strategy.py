#!/usr/bin/env python3
"""
12h_1d1w_donchian_volatility_breakout_v2
Hypothesis: Breakout from 1d Donchian channels with volatility filter on 12h chart.
- Long when price breaks above 20-day high + ATR expansion
- Short when price breaks below 20-day low + ATR expansion
- Use 1w trend filter to avoid counter-trend trades
- Volume confirmation to avoid false breakouts
- Designed for low trade frequency (15-25/year) to minimize fee drag
- Works in bull/bear via trend filter and volatility breakout logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d1w_donchian_volatility_breakout_v2"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high = np.full(len(close_1d), np.nan)
    donchian_low = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d ATR (14-period) for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1w EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
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
        atr_val = atr_1d_aligned[i]
        trend_up = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below Donchian low or volatility contracts
            if price < lower or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian high or volatility contracts
            if price > upper or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above Donchian high with volatility expansion and uptrend
            if price > upper and vol_ratio > 1.5 and atr_val > np.nanmedian(atr_1d_aligned[max(0, i-20):i]) and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: break below Donchian low with volatility expansion and downtrend
            elif price < lower and vol_ratio > 1.5 and atr_val > np.nanmedian(atr_1d_aligned[max(0, i-20):i]) and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals