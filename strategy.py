#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v1
# Strategy: 4h Donchian breakout with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum; volume confirmation filters false breakouts; 1d EMA50 ensures trend alignment. Works in bull via long breakouts, in bear via short breakdowns. Target: 25-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above prior high
        short_breakout = close[i] < donchian_low[i-1]  # Break below prior low
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if long_breakout and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and short_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and long_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals