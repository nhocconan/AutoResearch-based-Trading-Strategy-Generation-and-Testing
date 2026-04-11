#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Strategy: 1d Donchian breakout with weekly EMA trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Daily Donchian breakouts capture multi-day trends. Weekly EMA confirms higher timeframe trend direction.
# Volume > 1.5x 20-day average confirms institutional participation. Designed for low trade frequency (~10-25/year)
# to minimize fee drag. Works in bull markets via long breakouts and bear markets via short breakdowns.

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Donchian breakout signals
        breakout_up = high[i] > high_max_20[i-1]
        breakdown_down = low[i] < low_min_20[i-1]
        
        # Weekly EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Donchian breakout up AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Donchian breakdown down AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Donchian signal (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals