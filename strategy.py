#!/usr/bin/env python3
"""
1d_KAMA_Trend_Volume_Momentum
Hypothesis: KAMA direction + RSI + volume momentum filter on daily chart. KAMA adapts to volatility, RSI identifies momentum, volume confirms strength. Works in both bull and bear markets by filtering with trend and momentum.
"""

name = "1d_KAMA_Trend_Volume_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily close
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (Kaufman Adaptive Moving Average) ---
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder, will compute properly
    
    # Proper ER calculation
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    direction = np.abs(close_series - close_series.shift(10))
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI (14) ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # --- Volume momentum: volume > 20-day average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Weekly trend filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for KAMA, RSI, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume and RSI momentum
            if close[i] > kama[i] and rsi[i] > 55 and trend_up and vol_ok:
                # Long: price above KAMA, RSI > 55, weekly uptrend, volume momentum
                signals[i] = 0.25
                position = 1
            elif close[i] < kama[i] and rsi[i] < 45 and trend_down and vol_ok:
                # Short: price below KAMA, RSI < 45, weekly downtrend, volume momentum
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below KAMA or RSI < 40
                if close[i] < kama[i] or rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA or RSI > 60
                if close[i] > kama[i] or rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals