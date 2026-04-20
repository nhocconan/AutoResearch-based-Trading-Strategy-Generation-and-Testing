#!/usr/bin/env python3
"""
12h_Trend_Momentum_With_Volume_Confirmation
Hypothesis: Trade 12h momentum with daily trend filter and volume confirmation. 
Long when price > daily EMA50 + RSI > 50 + volume > 1.5x average volume.
Short when price < daily EMA50 + RSI < 50 + volume > 1.5x average volume.
Uses EMA for trend, RSI for momentum, volume filter to avoid false signals.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: daily filter avoids counter-trend trades, volume confirms conviction.
"""

name = "12h_Trend_Momentum_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema50_daily[i-1]
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = np.mean(gain[max(0, i-13):i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[max(0, i-13):i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume (20-period)
    avg_volume = np.zeros(n)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(volume[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: price > daily EMA50 + RSI > 50 + volume confirmation
            if close[i] > ema50_daily_aligned[i] and rsi[i] > 50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price < daily EMA50 + RSI < 50 + volume confirmation
            elif close[i] < ema50_daily_aligned[i] and rsi[i] < 50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < daily EMA50 OR RSI < 50
            if close[i] < ema50_daily_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > daily EMA50 OR RSI > 50
            if close[i] > ema50_daily_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals