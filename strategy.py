#!/usr/bin/env python3
"""
6h_Issue_30_Trend_HTF1dW1
Hypothesis: On 6h timeframe, combine Issue #30 (Donchian20 breakout + volume spike) with HTF trend from 1d and weekly EMA34 to capture strong momentum moves while avoiding counter-trend trades. Issue #30 provides precise entry timing with volume confirmation, while HTF EMA filters ensure alignment with higher timeframe trend. This reduces false breakouts and improves win rate in both bull and bear markets by only trading in direction of HTF trend. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 1w data ONCE before loop for HTF trend (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Issue #30 components on 6h
    # Donchian(20) breakout
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume spike (> 2x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # HTF trend filter (both 1d and 1w EMA34 must agree)
        uptrend = (close[i] > ema_34_1d_aligned[i]) and (close[i] > ema_34_1w_aligned[i])
        downtrend = (close[i] < ema_34_1d_aligned[i]) and (close[i] < ema_34_1w_aligned[i])
        
        # Long logic: Donchian breakout up + volume spike + HTF uptrend
        if close[i] > donchian_high[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Donchian breakout down + volume spike + HTF downtrend
        elif close[i] < donchian_low[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite Donchian breakout or HTF trend reversal
        elif position == 1 and (close[i] < donchian_low[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > donchian_high[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Issue_30_Trend_HTF1dW1"
timeframe = "6h"
leverage = 1.0