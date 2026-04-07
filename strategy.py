#!/usr/bin/env python3
"""
4h_donchian_breakout_1w_trend_volume_v2
Hypothesis: On 4h timeframe, buy when price breaks above 20-period Donchian high with volume confirmation and weekly uptrend, sell when breaks below 20-period Donchian low with volume confirmation and weekly downtrend. Uses weekly EMA40 for trend filter to avoid whipsaws in sideways markets. Designed for 15-30 trades/year to minimize fee drag while capturing major trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1w_trend_volume_v2"
timeframe = "4h"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema_40 = df_1w['close'].ewm(span=40, adjust=False).mean()
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40.values)
    
    # Donchian channels (20-period) on 4h data
    lookback = 20
    # Use pandas rolling for cleaner code with proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema_40_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_40_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_40_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume in weekly uptrend
            if (close[i] > donchian_high[i] and
                vol_confirm and 
                close[i] > ema_40_aligned[i]):  # weekly uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume in weekly downtrend
            elif (close[i] < donchian_low[i] and
                  vol_confirm and 
                  close[i] < ema_40_aligned[i]):  # weekly downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals