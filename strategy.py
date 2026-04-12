#!/usr/bin/env python3
"""
12h_1d_ma_crossover_volume_filter_v2
Hypothesis: 12-hour strategy using daily 50/200 EMA crossover for trend, with volume confirmation to avoid false breakouts.
Trades only in direction of daily trend when volume exceeds 1.5x 20-period average. Target: 20-30 trades/year (80-120 total) to minimize fee drift.
Works in bull/bear by only taking trend-aligned trades. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: 20-period average volume on 12h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > vol_ma[i] * 1.5
        
        # Trend determination
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Entry conditions
        if uptrend and volume_confirmed and position != 1:
            position = 1
            signals[i] = 0.25
        elif downtrend and volume_confirmed and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal
        elif position == 1 and downtrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and uptrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_ma_crossover_volume_filter_v2"
timeframe = "12h"
leverage = 1.0