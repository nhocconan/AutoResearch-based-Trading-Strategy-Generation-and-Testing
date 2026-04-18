#!/usr/bin/env python3
"""
1h_4h_1d_EMA_Trend_Filter
Hypothesis: In strong trends (EMA alignment across 1h/4h/1d), price pulls back to the 1h EMA21 and resumes trend.
Go long when 1h price crosses above EMA21 with 4h/1d uptrend confirmation; short when crosses below EMA21 with downtrend.
Uses volume filter to avoid low-volatility whipsaws. Designed for 1h timeframe with tight entries to limit trades.
Target: 15-35 trades/year (60-140 total over 4 years) to stay within fee limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # 4h EMA50 trend (tigher period for responsiveness)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA100 trend (stronger filter)
    df_1d = get_htf_data(prices, '1d')
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for EMA21 and HTF alignment
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21[i]) or np.isnan(volume_filter[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema21 = ema_21[i]
        vol_ok = volume_filter[i]
        ema4h = ema_50_4h_aligned[i]
        ema1d = ema_100_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above EMA21 with volume and 4h/1d uptrend
            if price > ema21 and vol_ok and ema4h > ema1d:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below EMA21 with volume and 4h/1d downtrend
            elif price < ema21 and vol_ok and ema4h < ema1d:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price crosses back below EMA21 or trend weakens
            if price < ema21 or ema4h < ema1d:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price crosses back above EMA21 or trend weakens
            if price > ema21 or ema4h > ema1d:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h_1d_EMA_Trend_Filter"
timeframe = "1h"
leverage = 1.0