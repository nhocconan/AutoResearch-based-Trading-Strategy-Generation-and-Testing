#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: On daily timeframe, use KAMA to determine trend direction (bullish/bearish) and RSI for entry timing.
In bullish trend (KAMA rising), go long when RSI crosses above 30 from below.
In bearish trend (KAMA falling), go short when RSI crosses below 70 from above.
Exit when RSI reaches opposite extreme (70 for long, 30 for short) or trend reverses.
Uses weekly timeframe for trend confirmation: only trade when weekly KAMA agrees with daily trend.
Designed for low trade frequency (10-20/year) by requiring multiple confluence factors.
Works in bull/bear via trend filter and mean-reversion exit at RSI extremes.
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
    
    # === DAILY KAMA (TREND) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = 0.0
        else:
            er[i] = change[i-9] / volatility[i-9] if volatility[i-9] > 0 else 0.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === DAILY RSI (ENTRY TIMING) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI(14)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === WEEKLY KAMA (TREND CONFIRMATION) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly Efficiency Ratio
    change_1w = np.abs(np.diff(close_1w, n=10))
    volatility_1w = np.sum(np.abs(np.diff(close_1w)), axis=1)
    er_1w = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if i < 10:
            er_1w[i] = 0.0
        else:
            er_1w[i] = change_1w[i-9] / volatility_1w[i-9] if volatility_1w[i-9] > 0 else 0.0
    
    # Weekly KAMA
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_1w = (er_1w * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama_1w = np.zeros(len(close_1w))
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align weekly KAMA to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend determination
        daily_trend_up = kama[i] > kama[i-1]
        daily_trend_down = kama[i] < kama[i-1]
        weekly_trend_up = kama_1w_aligned[i] > kama_1w_aligned[i-1]
        weekly_trend_down = kama_1w_aligned[i] < kama_1w_aligned[i-1]
        
        # Agreement between daily and weekly trend
        bullish_agreement = daily_trend_up and weekly_trend_up
        bearish_agreement = daily_trend_down and weekly_trend_down
        
        # Entry conditions
        long_entry = bullish_agreement and (rsi[i] > 30) and (rsi[i-1] <= 30)
        short_entry = bearish_agreement and (rsi[i] < 70) and (rsi[i-1] >= 70)
        
        # Exit conditions
        exit_long = (rsi[i] >= 70) or not bullish_agreement
        exit_short = (rsi[i] <= 30) or not bearish_agreement
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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