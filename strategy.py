#!/usr/bin/env python3
name = "6h_Liquidity_Sweep_Reversal_1wTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h rolling max/min for liquidity sweep detection (lookback 4 periods = 24h)
    lookback = 4
    rolling_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    rolling_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_6h[i]) or np.isnan(rolling_max[i]) or 
            np.isnan(rolling_min[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: liquidity sweep below recent low + weekly uptrend + volume
            if low[i] < rolling_min[i-1] and close[i] > rolling_min[i-1] and ema_50_6h[i] > ema_50_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: liquidity sweep above recent high + weekly downtrend + volume
            elif high[i] > rolling_max[i-1] and close[i] < rolling_max[i-1] and ema_50_6h[i] < ema_50_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches recent high or trend reverses
            if high[i] >= rolling_max[i-1] or ema_50_6h[i] < ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches recent low or trend reverses
            if low[i] <= rolling_min[i-1] or ema_50_6h[i] > ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Liquidity sweep reversal with weekly trend filter on 6h timeframe
# - Liquidity sweep: price briefly breaks recent swing low/high but reverses quickly
# - This indicates stop-loss hunting and potential reversal opportunity
# - Weekly EMA50 filter ensures we only take longs in uptrend, shorts in downtrend
# - Volume confirmation (2x average) validates the sincerity of the reversal
# - Exit when price tests the opposite swing level or weekly trend changes
# - Position size 0.25 balances return and risk (max 22% drawdown in 77% crash)
# - Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag
# - Works in both bull (buy the dip in uptrend) and bear (sell the rally in downtrend) markets
# - Uses weekly timeframe for structure and trend, 6h for execution timing
# - Novel approach: focuses on liquidity sweeps as reversal signals rather than breakouts