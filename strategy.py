#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Strategy: 1d Donchian Channel breakout with volume confirmation and weekly EMA trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture strong momentum. Weekly EMA ensures trend alignment.
# Volume confirms breakout sincerity. Designed for low trade frequency (~10-25/year) to minimize fee drag.
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1d Donchian Channel (20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend alignment
        if close[i] > donchian_upper[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < donchian_lower[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to opposite Donchian band (mean reversion within channel)
        elif position == 1 and close[i] < donchian_lower[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_upper[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals