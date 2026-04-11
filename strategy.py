#!/usr/bin/env python3
# 12h_1d_1w_donchian_breakout_volume_regime_v1
# Strategy: 12h Donchian(20) breakout with volume confirmation and 1d/1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum. Volume confirms breakout strength.
# 1d EMA50 and 1w EMA200 define trend regime to avoid counter-trend trades.
# Low frequency (~15-30/year) to minimize fee decay.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_donchian_breakout_volume_regime_v1"
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
    
    # 1w EMA(200) for regime filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Donchian(20) channels
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend regime: price above/below 1d EMA50 and 1w EMA200
        bull_regime = close[i] > ema_50_1d_aligned[i] and close[i] > ema_200_1w_aligned[i]
        bear_regime = close[i] < ema_50_1d_aligned[i] and close[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Donchian breakout + volume + regime alignment
        if close[i] > donchian_high[i-1] and vol_confirm[i] and bull_regime and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < donchian_low[i-1] and vol_confirm[i] and bear_regime and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: break of opposite Donchian band or regime change
        elif position == 1 and (close[i] < donchian_low[i] or not bull_regime):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or not bear_regime):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals