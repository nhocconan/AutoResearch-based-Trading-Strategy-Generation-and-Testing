#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime
Long when KAMA rising, RSI > 50, Chop > 61.8 (range)
Short when KAMA falling, RSI < 50, Chop > 61.8 (range)
Exit when KAMA direction changes or Chop < 38.2 (trend)
Uses weekly trend filter: only trade in direction of weekly KAMA
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on daily
    # Efficiency ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[9:] = change[9:] / np.maximum(volatility[9:], 1e-10)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppy Index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed ATR and true range sum
    atr_sum = np.zeros(n)
    tr_sum = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            atr_sum[i] = np.sum(atr[1:15])
            tr_sum[i] = np.sum(tr[1:15])
        else:
            atr_sum[i] = atr_sum[i-1] - atr_sum[i-1]/14 + atr[i]
            tr_sum[i] = tr_sum[i-1] - tr_sum[i-1]/14 + tr[i]
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / atr_sum[i]) / np.log10(14)
        else:
            chop[i] = 50
    
    # Weekly KAMA for trend filter
    close_weekly = df_weekly['close'].values
    change_w = np.abs(np.diff(close_weekly, n=10))
    volatility_w = np.sum(np.abs(np.diff(close_weekly)), axis=1)
    er_w = np.zeros(len(close_weekly))
    er_w[9:] = change_w[9:] / np.maximum(volatility_w[9:], 1e-10)
    sc_w = (er_w * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_w = np.full(len(close_weekly), np.nan)
    if len(close_weekly) > 9:
        kama_w[9] = close_weekly[9]
        for i in range(10, len(close_weekly)):
            kama_w[i] = kama_w[i-1] + sc_w[i] * (close_weekly[i] - kama_w[i-1])
    
    kama_w_aligned = align_htf_to_ltf(prices, df_weekly, kama_w)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    start_idx = max(30, 14)  # need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(kama_w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade in direction of weekly KAMA
        if i >= 1:
            weekly_rising = kama_w_aligned[i] > kama_w_aligned[i-1]
            weekly_falling = kama_w_aligned[i] < kama_w_aligned[i-1]
        else:
            weekly_rising = False
            weekly_falling = False
        
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop > 61.8 (range), weekly rising
            if kama_rising and rsi[i] > 50 and chop[i] > 61.8 and weekly_rising:
                signals[i] = size
                position = 1
            # Short: KAMA falling, RSI < 50, Chop > 61.8 (range), weekly falling
            elif kama_falling and rsi[i] < 50 and chop[i] > 61.8 and weekly_falling:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down OR Chop < 38.2 (trend)
            if not kama_rising or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turns up OR Chop < 38.2 (trend)
            if not kama_falling or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime_WeeklyTrend"
timeframe = "1d"
leverage = 1.0