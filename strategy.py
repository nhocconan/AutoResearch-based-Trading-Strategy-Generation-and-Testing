#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction,
filtered by RSI to avoid whipsaws, and enter only when price crosses KAMA with volume confirmation.
Exit when price reverts to KAMA. Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (10-20/year) by requiring multiple confluence factors.
Works in bull/bear via weekly trend filter and mean-reversion exit at KAMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_With_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === DAILY KAMA (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    abs_change = np.zeros(n)
    for i in range(10, n):
        abs_change[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.zeros(n)
    er[:10] = np.nan
    er[10:] = change[9:] / np.where(abs_change[10:] == 0, 1, abs_change[10:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros(n)
    kama[:10] = close[:10]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === DAILY RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # === DAILY VOLUME AVERAGE (20) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Weekly trend alignment: price above/below weekly EMA21
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > kama[i]) and (rsi[i] > 50) and vol_confirm and weekly_uptrend
        short_setup = (close[i] < kama[i]) and (rsi[i] < 50) and vol_confirm and weekly_downtrend
        
        # Exit conditions: mean reversion to KAMA
        exit_long = close[i] < kama[i]
        exit_short = close[i] > kama[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals