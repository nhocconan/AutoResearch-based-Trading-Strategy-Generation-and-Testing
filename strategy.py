#!/usr/bin/env python3
# 12h_1d_donchian_breakout_volume_v1
# Strategy: 12h Donchian channel breakout with 1d volume confirmation and 1d EMA trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture breakout moves. Volume > 1.5x 20-period average confirms institutional participation.
# EMA50 trend filter ensures trades align with higher timeframe trend. Designed for low trade frequency (~20-40/year) to minimize fee drag.
# Works in bull markets via upside breakouts and bear markets via downside breakouts with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA25 and EMA50 for trend filter
    ema_25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_25[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: EMA25 > EMA50 for uptrend, EMA25 < EMA50 for downtrend
        uptrend = ema_25[i] > ema_50[i]
        downtrend = ema_25[i] < ema_50[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Entry conditions
        # Long: Upside breakout AND uptrend AND volume confirmation
        if breakout_up and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Downside breakout AND downtrend AND volume confirmation
        elif breakout_down and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout or trend change
        elif position == 1 and (breakout_down or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals