#!/usr/bin/env python3
"""
1h_4h_DailyTrend_Breakout_Pullback
Hypothesis: Use 4h EMA trend direction and daily Donchian channels for breakout entries on 1h.
Only trade pullbacks to EMA in the direction of the 4h trend when price breaks
daily Donchian bands. This captures trend continuation with defined risk.
Works in bull/bear: trend filter avoids counter-trend trades, breakouts capture momentum.
Target: 20-40 trades/year via strict 4h trend + daily breakout + pullback confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h EMA for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === Daily Donchian channels for breakout levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    signals = np.zeros(n)
    
    # Warmup: covers 34 EMA and 20 Donchian
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine 4h trend
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Entry: only when flat
        if position == 0:
            # Long: uptrend + pullback to EMA + break above daily upper
            if uptrend and low[i] <= ema_4h_aligned[i] * 1.005 and high[i] > upper_20_aligned[i]:
                signals[i] = 0.20
                position = 1
                continue
            # Short: downtrend + pullback to EMA + break below daily lower
            elif downtrend and high[i] >= ema_4h_aligned[i] * 0.995 and low[i] < lower_20_aligned[i]:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit: trend reversal or opposite breakout
        elif position == 1:
            if not uptrend or low[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            if not downtrend or high[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_DailyTrend_Breakout_Pullback"
timeframe = "1h"
leverage = 1.0