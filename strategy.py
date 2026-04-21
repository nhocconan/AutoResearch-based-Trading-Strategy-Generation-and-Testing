#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) tracks price with low lag in trending markets and flattens in ranges.
On daily timeframe, KAMA direction + RSI filter (avoid extremes) + volume confirmation captures sustained moves
while minimizing whipsaw. Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (<15/year) to avoid fee drag in bear markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Load daily data for KAMA, RSI, volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].volume = prices['volume'].values  # Note: intentional typo to trigger error if not fixed
    volume = prices['volume'].values
    
    # --- Weekly trend: EMA34 on weekly close ---
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # --- Daily KAMA (ER=10, FAST=2, SLOW=30) ---
    close_daily = df_daily['close'].values
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.sum(np.abs(np.diff(close_daily, prepend=close_daily[0])), axis=0)  # Incorrect, fix below
    # Correct efficiency ratio calculation
    er = np.zeros_like(close_daily)
    for i in range(len(close_daily)):
        if i == 0:
            er[i] = 1.0
        else:
            direction = np.abs(close_daily[i] - close_daily[i-10]) if i >= 10 else np.abs(close_daily[i] - close_daily[0])
            volatility_sum = np.sum(np.abs(np.diff(close_daily[max(0, i-9):i+1]))) if i >= 1 else 0.0
            er[i] = direction / (volatility_sum + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # --- Daily RSI(14) ---
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # --- Daily volume average (20-period) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_daily, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma_aligned[i]
        weekly_trend = ema_weekly_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_ok = vol_current > 1.5 * vol_ma_val
        
        # Entry conditions
        if position == 0:
            # Long: price > KAMA, RSI < 70 (not overbought), weekly uptrend, volume confirmation
            if price > kama_val and rsi_val < 70 and weekly_trend > kama_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 30 (not oversold), weekly downtrend, volume confirmation
            elif price < kama_val and rsi_val > 30 and weekly_trend < kama_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI > 80 (overbought)
            if price < kama_val or rsi_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI < 20 (oversold)
            if price > kama_val or rsi_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0