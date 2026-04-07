#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses 12h Donchian channel breakout for entry, filtered by 1d EMA50 trend and volume surge.
# Exit when price reverses to opposite Donchian boundary or trend changes.
# Designed for low frequency (target: 12-37 trades/year) to minimize fee impact.
# Works in both bull/bear via trend filter: only long in uptrend, short in downtrend.

name = "12h_donchian20_1d_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channel (20 periods)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat, look for entry
            # Long breakout above upper Donchian with uptrend and volume
            if (high[i] > highest_high[i]) and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakdown below lower Donchian with downtrend and volume
            elif (low[i] < lowest_low[i]) and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        elif position == 1:  # Long position
            # Exit on trend reversal or price retracement to lower Donchian
            if not uptrend or (low[i] <= lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on trend reversal or price retracement to upper Donchian
            if not downtrend or (high[i] >= highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
    
    return signals