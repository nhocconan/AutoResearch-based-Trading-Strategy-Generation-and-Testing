#!/usr/bin/env python3
# 12h_1d_1w_donchian_breakout_volume_v1
# Strategy: 12h Donchian(20) breakout with volume confirmation and 1d/1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets.
# In bull markets, long breakouts above upper band with volume and 1d/1w uptrend.
# In bear markets, short breakouts below lower band with volume and 1d/1w downtrend.
# Uses 1d EMA50 and 1w EMA50 for trend filter to avoid counter-trend trades.
# Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_donchian_breakout_volume_v1"
timeframe = "12h"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Donchian channels for period 20
        lookback_start = max(0, i - 19)
        period_high = np.max(high[lookback_start:i+1])
        period_low = np.min(low[lookback_start:i+1])
        
        # Trend filters
        uptrend = close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 10-period average
        vol_lookback_start = max(0, i - 9)
        vol_avg_10 = np.mean(volume[vol_lookback_start:i+1]) if (i - vol_lookback_start + 1) >= 10 else 0
        vol_confirm = volume[i] > (1.5 * vol_avg_10) if vol_avg_10 > 0 else False
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if close[i] > period_high and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < period_low and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or opposite breakout
        elif position == 1 and (not uptrend or close[i] < period_low):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or close[i] > period_high):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals