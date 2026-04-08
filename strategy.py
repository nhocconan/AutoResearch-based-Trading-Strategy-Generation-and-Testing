#!/usr/bin/env python3
# 1d_donchian20_weekly_trend_volume_v1
# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation. Works in both bull and bear markets by using weekly trend direction to filter breakouts. Volume surge confirms institutional interest. Target: 15-25 trades/year per symbol to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian Channel (20)
    dc_period = 20
    dc_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Weekly trend: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = pd.Series(volume).rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly EMA or Donchian low
            if close[i] < ema_21_1w_aligned[i] or close[i] < dc_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly EMA or Donchian high
            if close[i] > ema_21_1w_aligned[i] or close[i] > dc_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian high, above weekly EMA, with volume surge
            if (close[i] > dc_high[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low, below weekly EMA, with volume surge
            elif (close[i] < dc_low[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals