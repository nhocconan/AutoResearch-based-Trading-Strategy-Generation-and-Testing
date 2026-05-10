#!/usr/bin/env python3
# 4H_Donchian_Breakout_Volume_Trend_12h - Optimized for BTC/ETH
# Hypothesis: Donchian channel breakouts with volume confirmation and 12h trend filter capture strong momentum moves while minimizing whipsaw.
# Uses 4h for entry/exit, 12h for trend filter, and volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency (~20-40/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull/bear markets by following 12h trend direction.

name = "4H_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h trend filter: EMA 34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        is_uptrend = close[i] > ema_34_12h_aligned[i]
        is_downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian upper + volume confirmation + 12h uptrend
            if close[i] > high_roll[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian lower + volume confirmation + 12h downtrend
            elif close[i] < low_roll[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below Donchian lower (mean reversion)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above Donchian upper (mean reversion)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals