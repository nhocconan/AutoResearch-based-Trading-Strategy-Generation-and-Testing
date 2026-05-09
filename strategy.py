#!/usr/bin/env python3
"""
1d_KAMA_Signal_WeeklyTrend_Volume
Hypothesis: Use KAMA to detect daily trend direction, confirmed by weekly trend (EMA34) and volume spikes.
Go long when KAMA is rising (bullish), weekly EMA34 up, and volume > 1.5x 20-day average.
Go short when KAMA is falling (bearish), weekly EMA34 down, and volume spike.
Exit when KAMA direction reverses or weekly trend flips.
KAMA adapts to market noise, reducing whipsaws in chop. Weekly trend filter ensures alignment with higher timeframe momentum.
Volume confirmation ensures trades have participation, reducing false breakouts.
Designed for low trade frequency (~10-25/year) with high win rate by requiring confluence.
Works in both bull and bear markets by following the weekly trend direction.
"""

name = "1d_KAMA_Signal_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SSC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    # KAMA = KAMA[1] + SSC * (Close - KAMA[1])
    fast_sc = 2 / (2 + 1)   # 2-day EMA
    slow_sc = 2 / (30 + 1)  # 30-day EMA
    
    change = np.abs(np.diff(close, k=10))  # |Close - Close[10]|
    # Pad change array for first 9 values
    change_padded = np.concatenate([np.full(9, np.nan), change])
    
    # Volatility: sum of absolute changes over 10 periods
    volatility = np.zeros(n)
    for i in range(9, n):
        volatility[i] = np.nansum(np.abs(np.diff(close[i-9:i+1], k=1)))
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility != 0
    er[mask] = change_padded[mask] / volatility[mask]
    er[~mask] = 0
    
    # Smoothness constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: rising if KAMA > KAMA[1], falling if KAMA < KAMA[1]
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    # Handle first element
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly trend: up if close > EMA34, down if close < EMA34
    weekly_trend_up = close > ema_34_1d
    weekly_trend_down = close < ema_34_1d
    
    # Volume filter: current volume > 1.5x 20-day average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for KAMA and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising + weekly uptrend + volume spike
            if kama_rising[i] and weekly_trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + weekly downtrend + volume spike
            elif kama_falling[i] and weekly_trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or weekly trend turns down
            if kama_falling[i] or not weekly_trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or weekly trend turns up
            if kama_rising[i] or not weekly_trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals