#!/usr/bin/env python3
# 1d_1w_donchian_breakout_v1
# Strategy: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Daily Donchian breakouts capture strong momentum. Weekly trend filter (EMA50)
# ensures we trade only in the direction of the higher timeframe trend. Volume confirmation
# (>1.5x 20-day average) filters out weak breakouts. Designed for low trade frequency
# (~15-25 trades/year) to minimize fee drag and work in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: close vs weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above Donchian high AND uptrend AND volume confirmation
        if high[i] > highest_high[i] and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below Donchian low AND downtrend AND volume confirmation
        elif low[i] < lowest_low[i] and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses back through the midpoint of the Donchian channel
        elif position == 1 and close[i] < (highest_high[i] + lowest_low[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (highest_high[i] + lowest_low[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals