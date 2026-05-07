#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Weekly trend filter: price above/below weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    trend_up = close > ema50_1w_aligned
    trend_down = close < ema50_1w_aligned
    
    # Daily Donchian(20) breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for weekly EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike and weekly uptrend
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike and weekly downtrend
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Donchian low or weekly trend turns down
            if close[i] < donchian_low_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Donchian high or weekly trend turns up
            if close[i] > donchian_high_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakouts on daily chart with weekly trend filter and volume spike capture strong institutional moves.
# Long when price breaks above 20-day high with volume confirmation in weekly uptrend.
# Short when price breaks below 20-day low with volume confirmation in weekly downtrend.
# Weekly trend filter ensures we only trade with the higher timeframe trend, reducing whipsaws.
# Volume spike (>2x average) ensures conviction behind the breakout.
# Designed for 6h timeframe to target 15-35 trades/year, avoiding overtrading.
# Works in bull markets (breaks above Donchian high in uptrend) and bear markets (breaks below Donchian low in downtrend).