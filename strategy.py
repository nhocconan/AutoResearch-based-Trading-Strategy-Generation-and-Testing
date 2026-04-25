#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: 6-hour Donchian(20) breakout with 1-week EMA50 trend filter and volume spike confirmation.
Captures strong momentum moves aligned with weekly trend while avoiding false breakouts in choppy/low-volume conditions.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull markets via trend-following breaks
and in bear markets via short breakouts when aligned with weekly downtrend. Volume spike ensures participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for volume average (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 6h Donchian(20) channels - calculated from LTF prices
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) + 1w EMA50 (50) + 1d vol MA (20)
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-day average volume
        volume_spike = curr_volume > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend alignment and volume confirmation
            # Long breakout: price breaks above Donchian high with uptrend and volume spike
            long_breakout = (curr_close > highest_high[i]) and uptrend and volume_spike
            # Short breakout: price breaks below Donchian low with downtrend and volume spike
            short_breakout = (curr_close < lowest_low[i]) and downtrend and volume_spike
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below Donchian low or trend changes to downtrend
            if curr_close < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian high or trend changes to uptrend
            if curr_close > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0