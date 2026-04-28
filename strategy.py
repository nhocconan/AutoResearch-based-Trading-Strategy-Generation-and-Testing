#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_RSI_Momentum_Volume
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) for trend direction, RSI for momentum, and volume confirmation to capture high-probability breakouts on 4h timeframe. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing strong trends. Designed for low trade frequency (15-40/year) to minimize fee drag while performing well in both bull and bear markets by following trend direction and momentum confirmation. Targets 60-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio: |close - close[10]| / sum(|diff| over 10 periods)
    change = np.abs(close_1d[10:] - close_1d[:-10])
    vol = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs correction - let's use proper loop
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 4h for momentum
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder's smoothing
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with 50 (neutral)
    rsi = np.concatenate([np.full(rsi_period, 50), rsi[:len(close)-rsi_period]]) if len(close) > rsi_period else np.full(len(close), 50)
    
    # Volume confirmation: >1.5x 24-period MA (1 day of 4h bars)
    vol_ma_24 = np.convolve(volume, np.ones(24)/24, mode='same')
    # Handle edges
    vol_ma_24[:12] = np.convolve(volume[:24], np.ones(12)/12, mode='valid') if len(volume) >= 24 else volume[:12]
    vol_ma_24[-12:] = np.convolve(volume[-24:], np.ones(12)/12, mode='valid') if len(volume) >= 24 else volume[-12:]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, rsi_period + 10)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Momentum: RSI > 55 for long, < 45 for short
        rsi_long = rsi[i] > 55
        rsi_short = rsi[i] < 45
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Entry conditions
        long_entry = uptrend and rsi_long and vol_confirm
        short_entry = downtrend and rsi_short and vol_confirm
        
        # Exit conditions: opposite signal or loss of momentum
        long_exit = not uptrend or rsi[i] < 50
        short_exit = not downtrend or rsi[i] > 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_Filter_RSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0