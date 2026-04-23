#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA34) and volume confirmation.
Long when price breaks above Donchian upper band AND 1d EMA34 up AND volume > 2.0x average.
Short when price breaks below Donchian lower band AND 1d EMA34 down AND volume > 2.0x average.
Exit when price touches the opposite Donchian band (mean reversion within channel).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-25 trades/year per symbol.
Donchian channels provide clear trend structure, daily EMA filter avoids counter-trend trades,
and volume confirmation ensures breakout validity. Works in both bull (breakouts) and bear (mean reversion in range).
"""

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
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_trend_up = close[i] > ema34_1d_aligned[i]  # using 12h close vs daily EMA
        daily_trend_down = close[i] < ema34_1d_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band AND daily uptrend AND volume confirmation
            if (high[i] > highest_high[i] and daily_trend_up and 
                vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band AND daily downtrend AND volume confirmation
            elif (low[i] < lowest_low[i] and daily_trend_down and 
                  vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches Donchian lower band (mean reversion)
                if low[i] <= lowest_low[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches Donchian upper band (mean reversion)
                if high[i] >= highest_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0