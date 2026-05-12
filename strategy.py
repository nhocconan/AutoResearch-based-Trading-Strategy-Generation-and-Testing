#!/usr/bin/env python3
# 160080: 4h_RCI_Momentum_Trend_Filter_Volume
# Hypothesis: RCI (Rank Correlation Index) detects short-term momentum extremes. Combined with 1-day EMA34 trend filter and volume confirmation, it captures high-probability reversals in both bull and bear markets. The RCI provides early momentum signals while the higher timeframe filter ensures alignment with the dominant trend, reducing false signals.

name = "4h_RCI_Momentum_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: first average is simple mean, then smoothed
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial averages
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # Wilder smoothing
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rci(close, period):
    """Calculate Rank Correlation Index (RCI)"""
    n = len(close)
    rci = np.full(n, np.nan)
    
    for i in range(period-1, n):
        # Get the window
        window = close[i-period+1:i+1]
        
        # Rank the prices (1 = lowest, period = highest)
        price_ranks = pd.Series(window).rank(method='average').values
        
        # Rank the time periods (1 = oldest, period = most recent)
        time_ranks = np.arange(1, period+1)
        
        # Calculate Spearman's rank correlation
        d_squared = np.sum((price_ranks - time_ranks) ** 2)
        rci_value = 1 - (6 * d_squared) / (period * (period**2 - 1))
        rci[i] = rci_value * 100  # Scale to -100 to 100
    
    return rci

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # RCI on 4h for momentum (9-period for sensitivity)
    rci = calculate_rci(close, 9)

    # Volume confirmation: >1.4x 20-period average (slightly higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rci[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RCI oversold (< -80) + price above EMA34 (uptrend) + volume confirmation
            if (rci[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RCI overbought (> 80) + price below EMA34 (downtrend) + volume confirmation
            elif (rci[i] > 80 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RCI overbought (> 60) or price below EMA34 (trend change)
            if (rci[i] > 60 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RCI oversold (< -60) or price above EMA34 (trend change)
            if (rci[i] < -60 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals