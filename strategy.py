#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA34 trend filter.
# Donchian breakouts capture momentum; volume confirms institutional participation;
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws.
# Works in bull/bear by filtering breakouts with daily trend. Targets 20-50 trades/year on 4h.

name = "4h_Donchian20_Breakout_Volume_1dEMA34_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, volume spike, price > 1d EMA34
            if (close[i] > highest_high[i] and 
                volume[i] > 1.8 * avg_volume[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, volume spike, price < 1d EMA34
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.8 * avg_volume[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian middle (mean reversion) OR breaks below lower band
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] < donchian_mid or 
                close[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian middle OR breaks above upper band
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] > donchian_mid or 
                close[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals