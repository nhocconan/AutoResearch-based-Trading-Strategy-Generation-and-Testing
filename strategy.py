#!/usr/bin/env python3
"""
6h_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian channel breakouts with weekly trend filter and volume capture
institutional moves. Weekly trend determines bias (long/short), Donchian(20) provides
entry/exit levels, volume confirms breakout strength. Works in bull (breakouts continue)
and bear (breakdowns continue) by trading with weekly trend. Targets 15-30 trades/year
by requiring confluence of Donchian breakout, volume spike, and weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1w_trend_volume_v1"
timeframe = "6h"
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
    
    # Weekly data for Donchian calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20): highest high and lowest low of past 20 weeks
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align to 6h timeframe (shift by 1 week for completed bars only)
    high_20w_6h = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_6h = align_htf_to_ltf(prices, df_1w, low_20w)
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average (6-period)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20w_6h[i]) or 
            np.isnan(low_20w_6h[i]) or 
            np.isnan(ema50_1w_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low OR trend turns down
            if close[i] < low_20w_6h[i] or close[i] < ema50_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high OR trend turns up
            if close[i] > high_20w_6h[i] or close[i] > ema50_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly Donchian high + volume + uptrend
            if (close[i] > high_20w_6h[i] and 
                vol_confirm and 
                close[i] > ema50_1w_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian low + volume + downtrend
            elif (close[i] < low_20w_6h[i] and 
                  vol_confirm and 
                  close[i] < ema50_1w_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals