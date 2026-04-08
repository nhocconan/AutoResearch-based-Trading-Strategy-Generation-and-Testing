#!/usr/bin/env python3
"""
12h_1d1w_rsi_momentum_v1
Hypothesis: RSI momentum with trend filter on 12h chart for low-frequency trading.
- Long when RSI(14) crosses above 50 on 12h + price above 1d EMA(50) + volume above average
- Short when RSI(14) crosses below 50 on 12h + price below 1d EMA(50) + volume above average
- Uses 1w trend filter to avoid counter-trend trades in extreme conditions
- Designed for low trade frequency (15-25/year) to minimize fee drag
- Works in bull/bear via trend filter and momentum confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d1w_rsi_momentum_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA (50-period) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate 1w EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Calculate RSI (14-period) on 12h
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema_1d = ema_50_1d_aligned[i]
        ema_1w = ema_50_1w_aligned[i]
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        
        # Determine trend alignment (both 1d and 1w agree)
        bullish_trend = price > ema_1d and ema_1d > ema_1w
        bearish_trend = price < ema_1d and ema_1d < ema_1w
        
        if position == 1:  # Long
            # Exit: RSI crosses below 50 or trend breaks down
            if rsi_now < 50 and rsi_prev >= 50 or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI crosses above 50 or trend breaks down
            if rsi_now > 50 and rsi_prev <= 50 or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI crosses above 50 with volume and bullish trend
            if rsi_now > 50 and rsi_prev <= 50 and vol_ratio > 1.3 and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI crosses below 50 with volume and bearish trend
            elif rsi_now < 50 and rsi_prev >= 50 and vol_ratio > 1.3 and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals