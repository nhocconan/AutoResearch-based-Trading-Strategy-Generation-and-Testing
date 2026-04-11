#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_regime_v1
# Strategy: 1d Donchian breakout with volume confirmation and weekly regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets. Volume confirms breakout strength. Weekly regime filter (price above/below weekly EMA200) ensures we trade with the higher timeframe trend. Low frequency (~10-20/year) minimizes fee drag. Works in bull markets via upside breakouts and in bear markets via downside breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Weekly EMA(200) for regime filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Donchian channels (20-period) on daily data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: price above/below weekly EMA200
        bull_regime = close[i] > ema_200_1w_aligned[i]
        bear_regime = close[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + regime alignment
        if (high[i] > high_max_20[i] and vol_confirm[i] and bull_regime and position != 1):
            position = 1
            signals[i] = 0.25
        elif (low[i] < low_min_20[i] and vol_confirm[i] and bear_regime and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or regime change
        elif position == 1 and (low[i] < low_min_20[i] or not bull_regime):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high[i] > high_max_20[i] or not bear_regime):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals