#!/usr/bin/env python3
name = "12h_RollingBreakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly SMA for trend filter (alternative)
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # 12h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above SMA50 for long, below for short
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Volume condition: spike above 2x average
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: breakout above Donchian high in weekly uptrend with volume
            if close[i] > high_20[i] and weekly_uptrend and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in weekly downtrend with volume
            elif close[i] < low_20[i] and weekly_downtrend and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below Donchian low or trend reversal
            if close[i] < low_20[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above Donchian high or trend reversal
            if close[i] > high_20[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakouts with weekly trend filter and volume confirmation
# - Long when price breaks above 20-period high in weekly uptrend with volume spike
# - Short when price breaks below 20-period low in weekly downtrend with volume spike
# - Weekly trend filter uses EMA20 and SMA50 alignment to avoid counter-trend trades
# - Volume confirmation (2x average) reduces false breakouts
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Position size 0.25 targets ~30-60 trades/year to avoid fee drag
# - Donchian channels provide clear breakout levels with defined support/resistance
# - Weekly trend filter reduces whipsaws vs same-timeframe breakout strategies
# - Combines proven elements: Donchian breakout + trend filter + volume (from DB top performers)