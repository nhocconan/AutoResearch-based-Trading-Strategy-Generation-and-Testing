#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and 1d volume confirmation.
- Long: RSI < 30 (oversold), 4h close > 4h EMA50 (uptrend), 1d volume > 1.5x 20-day average
- Short: RSI > 70 (overbought), 4h close < 4h EMA50 (downtrend), 1d volume > 1.5x 20-day average
- Exit: RSI crosses back to neutral zone (40 for long, 60 for short)
- Uses volume filter to avoid low-liquidity false signals.
Designed for 15-37 trades/year (60-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    if len(gain) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.zeros(len(close))
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    ema = np.full(len(close), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    
    for i in range(period, len(close)):
        ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI(14) on 1h
    rsi = calculate_rsi(close, 14)
    
    # Calculate EMA(50) on 4h
    ema_50_4h = calculate_ema(close_4h, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-day average volume on 1d
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need RSI, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5 * 20-day average volume (scaled)
        # Scale 1d average to 1h: approximate by dividing by 6 (24h/4h * 4h/1h = 6)
        vol_threshold = vol_ma_20_1d_aligned[i] * 1.5 / 6.0
        vol_confirmed = volume[i] > vol_threshold
        
        if position == 0:
            # Long: RSI < 30 (oversold), 4h close > 4h EMA50 (uptrend), volume confirmation
            if (rsi[i] < 30 and close[i] > ema_50_4h_aligned[i] and vol_confirmed):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought), 4h close < 4h EMA50 (downtrend), volume confirmation
            elif (rsi[i] > 70 and close[i] < ema_50_4h_aligned[i] and vol_confirmed):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back to 40 (neutral)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses back to 60 (neutral)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_1dVolume_MeanReversion"
timeframe = "1h"
leverage = 1.0