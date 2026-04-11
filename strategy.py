#!/usr/bin/env python3
# 1h_4d_rvol_breakout_v1
# Strategy: 1-hour Relative Volume (RVOL) breakout with 4-day trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Price breakouts above/below 1h Donchian channels (20-period) combined with
# RVOL > 1.5 (current volume > 1.5x 20-period average volume) capture momentum.
# The 4-day EMA(50) trend filter ensures trades align with higher timeframe
# direction, reducing false breakouts in sideways markets. Session filter (08-20 UTC)
# reduces noise. Designed for 15-37 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rvol_breakout_v1"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h Donchian channel (20-period) for breakout
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1h Relative Volume (RVOL): current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / (vol_avg_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(rvol[i]) or np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > donchian_high[i-1]  # Break above prior high
        bear_breakout = close[i] < donchian_low[i-1]   # Break below prior low
        
        # Volume confirmation: RVOL > 1.5
        vol_confirm = rvol[i] > 1.5
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Entry logic: breakout + volume + trend alignment
        if bull_breakout and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.20
        elif bear_breakout and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals