#!/usr/bin/env python3
# 1d_1w_ema_crossover_volume_trend_v1
# Hypothesis: Daily EMA crossover (21/50) with weekly EMA trend filter (200) and volume confirmation.
# Long: daily EMA21 > EMA50 AND weekly EMA200 rising AND volume > 1.5x 20-period average
# Short: daily EMA21 < EMA50 AND weekly EMA200 falling AND volume > 1.5x 20-period average
# Exit: opposite crossover or volume drops below average.
# Designed to capture strong trends with institutional alignment (weekly trend) and avoid whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_crossover_volume_trend_v1"
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
    
    # Daily EMA21 and EMA50
    ema21 = np.full(n, np.nan)
    ema50 = np.full(n, np.nan)
    
    # Calculate EMA21
    alpha21 = 2.0 / (21 + 1)
    ema21[20] = close[:21].mean()
    for i in range(21, n):
        ema21[i] = alpha21 * close[i] + (1 - alpha21) * ema21[i-1]
    
    # Calculate EMA50
    alpha50 = 2.0 / (50 + 1)
    ema50[49] = close[:50].mean()
    for i in range(50, n):
        ema50[i] = alpha50 * close[i] + (1 - alpha50) * ema50[i-1]
    
    # Weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    ema200_1w = np.full(len(df_1w), np.nan)
    alpha200 = 2.0 / (200 + 1)
    ema200_1w[199] = df_1w['close'].iloc[:200].mean()
    for i in range(200, len(df_1w)):
        ema200_1w[i] = alpha200 * df_1w['close'].iloc[i] + (1 - alpha200) * ema200_1w[i-1]
    
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = volume[i-20:i].mean()
    
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        ema21_val = ema21[i]
        ema50_val = ema50[i]
        ema200_val = ema200_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if np.isnan(ema21_val) or np.isnan(ema50_val) or np.isnan(ema200_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if ema21_val <= ema50_val or ema200_val < ema200_1w_aligned[i-1] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if ema21_val >= ema50_val or ema200_val > ema200_1w_aligned[i-1] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if ema21_val > ema50_val and ema200_val > ema200_1w_aligned[i-1] and vol_filter:
                position = 1
                signals[i] = 0.25
            elif ema21_val < ema50_val and ema200_val < ema200_1w_aligned[i-1] and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals