#!/usr/bin/env python3
"""
6h_1d1w_volatility_breakout_v2
Hypothesis: 6-hour strategy using daily and weekly volatility breakouts with volume confirmation.
Long when price breaks above 6h Donchian(20) high + daily ATR expansion + weekly trend filter.
Short when price breaks below 6h Donchian(20) low + daily ATR expansion + weekly trend filter.
Exit when price crosses opposite Donchian boundary or ATR contraction.
Uses discrete position sizing (0.25) to manage risk. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_volatility_breakout_v2"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.full_like(high, np.nan, dtype=float)
    if period == 1:
        atr = tr
    else:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float), np.full_like(high, np.nan, dtype=float)
    
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(high, np.nan, dtype=float)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate daily ATR moving average for expansion detection
    atr_ma_1d = np.full_like(atr_1d, np.nan, dtype=float)
    for i in range(10, len(atr_1d)):
        atr_ma_1d[i] = np.mean(atr_1d[i-10:i])
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Calculate 6h Donchian channels
    donch_period = 20
    donch_upper, donch_lower = calculate_donchian(high, low, donch_period)
    
    # Align indicators to 6-hour timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        atr = atr_1d_aligned[i]
        atr_ma = atr_ma_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        
        # Volatility expansion condition
        vol_expansion = atr > atr_ma * 1.2
        
        if position == 1:  # Long
            # Exit: price breaks below lower Donchian or volatility contracts
            if price < lower or (vol_ratio < 1.0 and atr < atr_ma):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above upper Donchian or volatility contracts
            if price > upper or (vol_ratio < 1.0 and atr < atr_ma):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper Donchian with vol expansion and weekly uptrend
            if price > upper and vol_expansion and vol_ratio > 1.5 and price > ema_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower Donchian with vol expansion and weekly downtrend
            elif price < lower and vol_expansion and vol_ratio > 1.5 and price < ema_trend:
                position = -1
                signals[i] = -0.25
    
    return signals