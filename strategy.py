#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 12h timeframe, enter long when price breaks above Donchian(20) high with 1d EMA uptrend and volume confirmation; enter short when price breaks below Donchian(20) low with 1d EMA downtrend and volume confirmation. Exit on opposite Donchian break or trend reversal. Uses trend-following breakouts with volume filter to avoid false signals, designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets. Targets 50-150 total trades over 4 years (12-37/year) for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA on daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h timeframe
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns down
            if close[i] < donchian_low[i] or not above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns up
            if close[i] > donchian_high[i] or not below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above Donchian high with uptrend - go long
                if close[i] > donchian_high[i] and above_ema:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below Donchian low with downtrend - go short
                elif close[i] < donchian_low[i] and below_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals