#!/usr/bin/env python3
"""
6h_KAMA_Trend_RSI_Filter_Volume
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter.
Combined with RSI extremes for mean reversion in ranging markets and volume confirmation for breakout strength.
Works in both bull and bear markets by adapting trend sensitivity and using RSI for reversals.
Targets 20-30 trades/year to minimize fee drag.
"""

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
    
    # Get daily data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # Efficiency ratio: |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d))
    abs_change = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
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
    
    # Calculate RSI on daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first 14 values as NaN (not enough data)
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA
        trend_up = close[i] > kama_aligned[i]
        trend_down = close[i] < kama_aligned[i]
        
        # RSI conditions: oversold (<30) or overbought (>70)
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volume confirmation: >1.5x 20-period MA
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic
        long_entry = vol_confirm and rsi_oversold and trend_up
        short_entry = vol_confirm and rsi_overbought and trend_down
        
        # Exit logic: RSI returns to neutral range or trend reversal
        long_exit = (rsi_aligned[i] > 50) or (not trend_up)
        short_exit = (rsi_aligned[i] < 50) or (not trend_down)
        
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

name = "6h_KAMA_Trend_RSI_Filter_Volume"
timeframe = "6h"
leverage = 1.0