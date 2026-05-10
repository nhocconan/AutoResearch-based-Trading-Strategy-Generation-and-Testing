#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Trend
# Hypothesis: 12h Donchian channel breakouts with volume confirmation and 1d EMA trend filter capture major moves while minimizing false signals in ranging markets.
# The 12h timeframe reduces trade frequency to avoid fee drag, while volume confirmation ensures breakouts have participation.
# The 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, improving win rate in both bull and bear markets.
# Designed for 15-25 trades/year to minimize fee drag and maximize edge.

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # Donchian Channel (20-period)
    dc_period = 20
    # Upper band: highest high over last 20 periods
    dc_upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    # Lower band: lowest low over last 20 periods
    dc_lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # 1d EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above upper DC AND volume confirmation AND 1d EMA uptrend
            if close[i] > dc_upper[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower DC AND volume confirmation AND 1d EMA downtrend
            elif close[i] < dc_lower[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower DC OR loss of volume confirmation
            if close[i] < dc_lower[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper DC OR loss of volume confirmation
            if close[i] > dc_upper[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals