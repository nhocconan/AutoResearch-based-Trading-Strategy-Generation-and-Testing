#!/usr/bin/env python3
"""
1d_KAMA_StochRSI_Trend_v1
Strategy: KAMA (adaptive trend) + Stochastic RSI (mean reversion) on daily timeframe.
Long: KAMA trending up + StochRSI oversold (< 0.2)
Short: KAMA trending down + StochRSI overbought (> 0.8)
Exit: Opposite signal or StochRSI crosses back to neutral zone (0.2-0.8)
Designed for 1d timeframe: ~10-25 trades/year per symbol (40-100 total over 4 years).
Uses 1w timeframe for trend confirmation (KAMA slope).
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
    
    # Get daily data for KAMA and StochRSI
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    change_t = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_t = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            num = np.sum(change_t[max(0, i-9):i+1])  # 10-period change
            den = np.sum(volatility_t[max(0, i-9):i+1])  # 10-period volatility
            er[i] = num / den if den != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA slope (trend)
    kama_slope = np.diff(kama, prepend=0)
    
    # === STOCHASTIC RSI ===
    # RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_high = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    rsi_low = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
    
    # === WEEKLY TREND FILTER (KAMA) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly KAMA
    change_w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_w = np.abs(np.diff(close_1w))
    er_w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            er_w[i] = 0
        else:
            num = np.sum(change_w[max(0, i-9):i+1])
            den = np.sum(volatility_w[max(0, i-9):i+1])
            er_w[i] = num / den if den != 0 else 0
    
    fast_sc_w = 2 / (2 + 1)
    slow_sc_w = 2 / (30 + 1)
    sc_w = (er_w * (fast_sc_w - slow_sc_w) + slow_sc_w) ** 2
    
    kama_w = np.zeros_like(close_1w)
    kama_w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_w[i] = kama_w[i-1] + sc_w[i] * (close_1w[i] - kama_w[i-1])
    
    # Weekly trend: price above/below weekly KAMA
    weekly_uptrend = close_1w > kama_w
    weekly_downtrend = close_1w < kama_w
    
    # Align daily indicators to hourly? No, we're on 1d timeframe
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for StochRSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(stoch_rsi[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_setup = (kama_slope[i] > 0 and 
                     stoch_rsi[i] < 0.2 and 
                     weekly_uptrend_aligned[i] > 0.5)
        short_setup = (kama_slope[i] < 0 and 
                      stoch_rsi[i] > 0.8 and 
                      weekly_downtrend_aligned[i] > 0.5)
        
        # Exit conditions
        long_exit = (kama_slope[i] < 0 or 
                    stoch_rsi[i] > 0.8)
        short_exit = (kama_slope[i] > 0 or 
                     stoch_rsi[i] < 0.2)
        
        if position == 0:
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_StochRSI_Trend_v1"
timeframe = "1d"
leverage = 1.0