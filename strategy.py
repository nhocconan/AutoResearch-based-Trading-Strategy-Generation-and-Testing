#!/usr/bin/env python3
"""
1d_keltner_rsi_v1
Hypothesis: Keltner channel (EMA-based) breakout with RSI momentum and volume filter for 1d timeframe.
- Long: Close > Upper Keltner + RSI > 55 + volume > 1.5x 20-day avg
- Short: Close < Lower Keltner + RSI < 45 + volume > 1.5x 20-day avg
- Exit: Close crosses back below/above EMA(20) or volume drops below average
- Uses 1w EMA200 trend filter: only long when price > weekly EMA200, short when price < weekly EMA200
- Designed for 10-20 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_rsi_v1"
timeframe = "1d"
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

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr = np.full_like(close, np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = calculate_ema(close_1w, 200)
    
    # Calculate daily EMA20 for Keltner center
    ema_20 = calculate_ema(close, 20)
    
    # Calculate ATR(10) for Keltner width
    atr_10 = calculate_atr(high, low, close, 10)
    
    # Calculate Keltner bands
    keltner_upper = ema_20 + (2 * atr_10)
    keltner_lower = ema_20 - (2 * atr_10)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA200 to daily
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        rsi_val = rsi[i]
        trend_filter = ema_200_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below EMA20 or volume drops below average
            if price < ema_20[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above EMA20 or volume drops below average
            if price > ema_20[i] or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper Keltner with RSI > 55, volume expansion, and uptrend on weekly
            if price > upper and rsi_val > 55 and vol_ratio > 1.5 and price > trend_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower Keltner with RSI < 45, volume expansion, and downtrend on weekly
            elif price < lower and rsi_val < 45 and vol_ratio > 1.5 and price < trend_filter:
                position = -1
                signals[i] = -0.25
    
    return signals