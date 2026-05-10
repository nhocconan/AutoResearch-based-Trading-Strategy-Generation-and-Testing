#!/usr/bin/env python3
# 1D_KAMA_Trend_RSI_Volume
# Hypothesis: Trade in direction of daily KAMA trend with RSI momentum and volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.
# Long: KAMA upward, RSI > 55, volume > 1.3x average.
# Short: KAMA downward, RSI < 45, volume > 1.3x average.
# Works in bull/bear by following adaptive trend and using momentum/volume filters.
# Target: 15-25 trades/year per symbol.

name = "1D_KAMA_Trend_RSI_Volume"
timeframe = "1d"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period
    close_s = pd.Series(close)
    # Efficiency ratio
    change = abs(close - np.roll(close, 10))
    change[:10] = 0  # first 10 values invalid
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_uptrend = close_1w > sma50_1w
    weekly_downtrend = close_1w < sma50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 0:
            # Enter long: weekly uptrend + KAMA up + RSI > 55 + volume
            if weekly_up and kama_up and rsi[i] > 55 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + KAMA down + RSI < 45 + volume
            elif weekly_down and kama_down and rsi[i] < 45 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA turns down or RSI < 50
            if not kama_up or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up or RSI > 50
            if not kama_down or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals