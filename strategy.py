#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Contrarian_Reversal_ZScore_Pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d Close for Z-score calculation ===
    close_1d = df_1d['close'].values
    
    # === 1d Z-score of close (20-day) ===
    mean_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    zscore = (close_1d - mean_20) / (std_20 + 1e-10)
    zscore_4h = align_htf_to_ltf(prices, df_1d, zscore)
    
    # === 1w EMA50 for long-term trend filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 1d Previous day's pivot points (HLC/3) ===
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Z-score and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(zscore_4h[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Z-score < -2.0 (oversold) + above S1 + long-term uptrend + volume
            long_cond = (zscore_4h[i] < -2.0 and 
                        close[i] > s1_4h[i] and 
                        close[i] > ema50_1w_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: Z-score > 2.0 (overbought) + below R1 + long-term downtrend + volume
            short_cond = (zscore_4h[i] > 2.0 and 
                         close[i] < r1_4h[i] and 
                         close[i] < ema50_1w_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Z-score > -0.5 (mean reversion) or breakdown below S1
            exit_cond = (zscore_4h[i] > -0.5 or close[i] < s1_4h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Z-score < 0.5 (mean reversion) or breakout above R1
            exit_cond = (zscore_4h[i] < 0.5 or close[i] > r1_4h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Contrarian reversal strategy using 1d Z-score (20) for extreme
# deviations combined with 1w EMA50 trend filter and 1d pivot levels (S1/R1) for
# entry/exit. Enters long when price is significantly oversold (Z<-2) above S1
# in a long-term uptrend, and short when significantly overbought (Z>2) below R1
# in a long-term downtrend. Uses volume confirmation and exits on mean reversion
# (Z-score > -0.5 for longs, < 0.5 for shorts) or pivot breakdown/breakout.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in
# downtrend) markets. Targets 20-60 trades over 4 years (5-15/year) to minimize
# fee drag. Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via
# statistical extremes and institutional pivot levels.