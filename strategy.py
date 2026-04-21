#!/usr/bin/env python3
"""
12h_RangeReversal_Bollinger_3std_RSI2_Extreme
Hypothesis: Mean reversion at extreme Bollinger Bands (3 std) with RSI(2) <5 or >95.
Works in both bull and bear markets by buying capitulation and selling euphoria.
Uses 1d trend filter (price > EMA50) to avoid counter-trend entries.
Designed for 12h timeframe to limit trades to ~15-30/year.
"""

import numpy as np
import pandas as pd
from mtrand import rand

# Simple RSI implementation to avoid external dependencies
def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, 50.0)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Wilder's smoothing
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    # Set first period values to 50 (neutral)
    rsi[:period] = 50.0
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema50_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * multiplier + ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (3 std dev) on 12h close
    bb_length = 20
    bb_std = 3.0
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    close_series = pd.Series(prices['close'].values)
    if len(close_series) >= bb_length:
        sma = close_series.rolling(window=bb_length, min_periods=bb_length).mean().values
        std_dev = close_series.rolling(window=bb_length, min_periods=bb_length).std().values
        upper_band = sma + (bb_std * std_dev)
        lower_band = sma - (bb_std * std_dev)
    
    # RSI(2) for extreme readings
    rsi2 = calculate_rsi(prices['close'].values, period=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(sma[i]) or np.isnan(std_dev[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 00-23 UTC (all hours for 12h)
        # No session filter for 12h to capture major moves
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: price at lower Bollinger Band (3 std) + RSI2 oversold + uptrend
            if (price <= lower_band[i] and 
                rsi2[i] < 5 and  # extremely oversold
                price > ema50_1d_aligned[i] and  # uptrend filter
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price at upper Bollinger Band (3 std) + RSI2 overbought + downtrend
            elif (price >= upper_band[i] and 
                  rsi2[i] > 95 and  # extremely overbought
                  price < ema50_1d_aligned[i] and  # downtrend filter
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above SMA or RSI2 > 60
            if price > sma[i] or rsi2[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below SMA or RSI2 < 40
            if price < sma[i] or rsi2[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RangeReversal_Bollinger_3std_RSI2_Extreme"
timeframe = "12h"
leverage = 1.0