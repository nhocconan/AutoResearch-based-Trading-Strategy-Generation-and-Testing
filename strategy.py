#!/usr/bin/env python3
# 4h_RSI_4hTrend_Volume
# Hypothesis: Use 4h RSI(14) with overbought/oversold levels and 4h EMA50 trend filter.
# Long when RSI < 30 and price > EMA50; short when RSI > 70 and price < EMA50.
# Volume confirmation requires current volume > 20-period average.
# Exits on RSI crossing back to neutral zone (40-60) or trend failure.
# Designed for low frequency (20-40 trades/year) to avoid fee drag. Works in bull (catch oversold bounces)
# and bear (catch overbought reversals) with trend filter and volume confirmation.

name = "4h_RSI_4hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI on 4h data
    rsi_vals = rsi(close, 14)
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi_vals[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema_50[i]
        trend_down = close[i] < ema_50[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # RSI conditions
        rsi_oversold = rsi_vals[i] < 30
        rsi_overbought = rsi_vals[i] > 70
        rsi_neutral = (rsi_vals[i] >= 40) & (rsi_vals[i] <= 60)
        
        if position == 0:
            # LONG: RSI oversold, price above EMA50, volume confirmation
            if rsi_oversold and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought, price below EMA50, volume confirmation
            elif rsi_overbought and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or trend fails
            if rsi_neutral or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or trend fails
            if rsi_neutral or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals