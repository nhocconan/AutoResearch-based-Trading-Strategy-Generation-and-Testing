#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_Volume
Hypothesis: KAMA trend on daily timeframe with volume confirmation.
Long when KAMA rising and volume > 1.5x average; short when KAMA falling and volume > 1.5x average.
Use weekly timeframe for trend filter: only trade long when price > weekly EMA50, short when price < weekly EMA50.
Exit when KAMA changes direction or volume drops.
Designed for 1d timeframe to capture medium-term trends with low trade frequency.
Target: 20-50 trades over 4 years (5-12/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === KAMA calculation on daily close ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === Weekly EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # KAMA direction: rising if current > previous, falling if current < previous
        if i > 0:
            kama_prev = kama_aligned[i-1]
            kama_rising = kama_val > kama_prev
            kama_falling = kama_val < kama_prev
        else:
            kama_rising = False
            kama_falling = False
        
        if position == 0:
            # Long: KAMA rising + price above weekly EMA50 + volume confirmation
            if (kama_rising and
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + price below weekly EMA50 + volume confirmation
            elif (kama_falling and
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when KAMA changes direction or volume drops
            if position == 1:
                if not kama_rising or vol_ratio_val < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not kama_falling or vol_ratio_val < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Trend_With_Volume"
timeframe = "1d"
leverage = 1.0