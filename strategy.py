#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Breakout_Trend_Filter
Hypothesis: Daily Camarilla pivot levels provide institutional-grade support/resistance.
Breakouts above H4 or below L4 with volume expansion indicate strong momentum.
Weekly trend filter (price above/below weekly EMA20) ensures alignment with higher timeframe trend,
reducing whipsaws in choppy markets. Targets 15-30 trades/year on 12h timeframe.
Works in bull markets via momentum continuation and bear markets via trend-following short signals.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align daily Camarilla levels to 12h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above H4 with volume expansion
        # 2. Price above weekly EMA20 for uptrend alignment
        breakout_long = (close[i] > camarilla_h4_aligned[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_20_1w_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below L4 with volume expansion
        # 2. Price below weekly EMA20 for downtrend alignment
        breakdown_short = (close[i] < camarilla_l4_aligned[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_20_1w_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_1w_Camarilla_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0