#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index for regime filter.
Enters long when KAMA is rising, RSI > 50, and market is trending (CHOP < 38.2).
Enters short when KAMA is falling, RSI < 50, and market is trending (CHOP < 38.2).
Uses weekly timeframe trend filter (EMA34) to avoid counter-trend trades in strong weekly trends.
Designed for 30-100 total trades over 4 years (7-25/year) with discrete position sizing (0.0, ±0.25).
Works in both bull and bear markets by only trading in the direction of the weekly trend
and avoiding ranging markets via choppiness filter.
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
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC) for KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, np.where(kama[1:] < kama[:-1], -1, 0))
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # align with close
    
    # Choppiness Index (CHOP) - 14 period
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr[1:] = tr
    
    # True Range for first period
    tr[0] = high[0] - low[0]
    atr[0] = tr[0]
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14), 
                    50)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly trend: 1 if close > EMA34, -1 if close < EMA34
    weekly_trend = np.where(close_1w > ema_34_1w, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Volume confirmation: volume > 1.5 * 20-day EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup periods
    start_idx = max(14, 34)  # RSI and weekly EMA need 34 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: KAMA rising, RSI > 50, CHOP < 38.2 (trending), volume spike, weekly trend bullish
        if (kama_dir[i] == 1 and rsi[i] > 50 and chop[i] < 38.2 and 
            volume_spike[i] and weekly_trend_aligned[i] == 1):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: KAMA falling, RSI < 50, CHOP < 38.2 (trending), volume spike, weekly trend bearish
        elif (kama_dir[i] == -1 and rsi[i] < 50 and chop[i] < 38.2 and 
              volume_spike[i] and weekly_trend_aligned[i] == -1):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite KAMA direction or chop becomes too high (ranging market)
        elif position == 1 and (kama_dir[i] == -1 or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (kama_dir[i] == 1 or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0