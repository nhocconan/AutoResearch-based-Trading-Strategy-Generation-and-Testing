#!/usr/bin/env python3
"""
Hypothesis: 1-day RSI with weekly trend filter and volume confirmation.
In oversold conditions (RSI < 30) with weekly uptrend: long.
In overbought conditions (RSI > 70) with weekly downtrend: short.
RSI identifies reversal points, weekly trend filters for direction,
volume confirms participation. Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=np.float64)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full_like(close, np.nan, dtype=np.float64)
    rs[period:] = avg_gain[period:] / np.where(avg_loss[period:] == 0, 1e-10, avg_loss[period:])
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend
    wk_close = df_1w['close'].values
    ema_34_1w = np.empty_like(wk_close, dtype=np.float64)
    ema_34_1w.fill(np.nan)
    if len(wk_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1w[33] = np.mean(wk_close[:34])
        for i in range(34, len(wk_close)):
            ema_34_1w[i] = alpha * wk_close[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.empty_like(vol_1d, dtype=np.float64)
    vol_ma_20_1d.fill(np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate daily RSI
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI (14) + weekly EMA (34) + daily volume MA (20)
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        rsi_val = rsi[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Weekly close price for trend comparison
        wk_close_price = df_1w['close'].values
        wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close_price)
        if np.isnan(wk_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_close = wk_close_aligned[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        if position == 0:
            # Oversold with weekly uptrend: long
            if rsi_val < 30 and weekly_close > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Overbought with weekly downtrend: short
            elif rsi_val > 70 and weekly_close < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought or trend change
            if rsi_val > 70 or weekly_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: oversold or trend change
            if rsi_val < 30 or weekly_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0