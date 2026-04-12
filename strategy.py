#!/usr/bin/env python3
"""
12h_1d_momentum_follow
Combines 1-day momentum (price above/below 20-day SMA) with 12-hour RSI pullback entries.
Long when: 1d close > SMA20 AND 12h RSI < 40 (pullback in uptrend)
Short when: 1d close < SMA20 AND 12h RSI > 60 (pullback in downtrend)
Requires volume > 1.5x 20-period average for confirmation.
Exits when 12h RSI crosses 50 (momentum fade) or trend reverses.
Designed for low trade frequency (~20-30 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the higher timeframe trend.
"""

name = "12h_1d_momentum_follow"
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
    
    # Get daily data for trend determination
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day 20-period SMA for trend filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # 12-hour RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align 1d trend to 12h timeframe
    trend_up = sma20_1d > 0  # placeholder for alignment
    trend_down = sma20_1d > 0  # placeholder
    
    # Actually compute and align the trend signals
    trend_raw = np.where(close_1d > sma20_1d, 1, np.where(close_1d < sma20_1d, -1, 0))
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if trend data not ready
        if np.isnan(trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_aligned[i]
        
        # Long entry: uptrend on 1d + RSI pullback (oversold) + volume
        if (trend == 1 and rsi[i] < 40 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: downtrend on 1d + RSI pullback (overbought) + volume
        elif (trend == -1 and rsi[i] > 60 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: RSI crosses 50 (momentum fade) or trend reverses
        elif position == 1 and (rsi[i] > 50 or trend != 1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 50 or trend != -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals