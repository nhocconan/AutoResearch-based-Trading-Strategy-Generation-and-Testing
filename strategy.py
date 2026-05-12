#!/usr/bin/env python3
# 6h_Donchian_20_Breakout_1dTrend_Volume_Confirmation
# Hypothesis: Donchian(20) breakout on 6h with 1d trend filter (price above/below 50 EMA) and volume confirmation.
# This structure has shown strong performance across multiple timeframes in prior experiments.
# The 1d EMA50 provides a robust trend filter, while volume confirmation ensures breakouts have conviction.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 50-150 total trades over 4 years (~12-37/year).

name = "6h_Donchian_20_Breakout_1dTrend_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Donchian Channel (20) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = high[i] > high_20[i-1]  # Current high breaks above previous period's high
        breakout_down = low[i] < low_20[i-1]  # Current low breaks below previous period's low
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: upward breakout, price above 1d EMA50, volume confirmation
            if breakout_up and price_above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: downward breakout, price below 1d EMA50, volume confirmation
            elif breakout_down and price_below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: downward breakout or price falls below 1d EMA50
            if breakout_down or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: upward breakout or price rises above 1d EMA50
            if breakout_up or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals