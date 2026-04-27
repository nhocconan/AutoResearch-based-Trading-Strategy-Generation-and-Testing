#!/usr/bin/env python3
"""
Hypothesis: 1-hour momentum with 4-hour trend filter and volume confirmation.
In bull market (price > 4-hour EMA50): long when RSI crosses above 50 and volume > 1.5x average.
In bear market (price < 4-hour EMA50): short when RSI crosses below 50 and volume > 1.5x average.
Uses 4-hour trend for direction and 1-hour RSI for timing to avoid false signals.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for trend filter and volume average
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = np.empty_like(close_4h, dtype=np.float64)
    ema_50_4h.fill(np.nan)
    if len(close_4h) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4-hour volume moving average (20-period)
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = np.empty_like(volume_4h, dtype=np.float64)
    vol_ma_20_4h.fill(np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate 1-hour RSI
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need RSI (14+1=15), EMA50 (50), volume MA (20)
    start_idx = max(15, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        volume_now = volume[i]
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        ema_trend = ema_50_4h_aligned[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        
        # Volume filter: volume > 1.5x 4-hour average
        vol_filter = volume_now > 1.5 * vol_ma
        
        # RSI crossing conditions
        rsi_cross_up = rsi_prev < 50 and rsi_now >= 50
        rsi_cross_down = rsi_prev > 50 and rsi_now <= 50
        
        if position == 0:
            # Bull market (price > 4-hour EMA50): look for long
            if price_now > ema_trend and rsi_cross_up and vol_filter:
                signals[i] = size
                position = 1
            # Bear market (price < 4-hour EMA50): look for short
            elif price_now < ema_trend and rsi_cross_down and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50 or trend change to bear
            if rsi_cross_down or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses above 50 or trend change to bull
            if rsi_cross_up or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0