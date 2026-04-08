#!/usr/bin/env python3
# 4h_Donchian_breakout_1d_trend_volume_v1
# Hypothesis: Trade Donchian(20) breakouts on 4h in direction of 1d trend with volume confirmation.
# Works in bull/bear markets by aligning with higher timeframe trend. Volume reduces false breakouts.
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "4h_Donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d trend direction: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    for i in range(donchian_len, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Long: price breaks above Donchian high + 1d uptrend + volume confirmation
        if (close[i] > highest_high[i] and 
            trend_1d_aligned[i] == 1 and 
            vol_confirm[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + 1d downtrend + volume confirmation
        elif (close[i] < lowest_low[i] and 
              trend_1d_aligned[i] == -1 and 
              vol_confirm[i]):
            signals[i] = -0.25
    
    return signals