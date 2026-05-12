#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND
# Hypothesis: 4-hour Donchian channel breakout with volume confirmation and 1-day trend filter.
# Long when price breaks above 4h Donchian upper band with volume above average and price above 1d EMA.
# Short when price breaks below 4h Donchian lower band with volume above average and price below 1d EMA.
# Exit when price returns to opposite Donchian band or trend invalidates.
# Designed for low-frequency, high-probability setups targeting 20-40 trades/year to minimize fee drag.

name = "4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND"
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
    
    # 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        volume_confirmed = volume[i] > volume_ma[i]
        
        if position == 0:
            # LONG: Break above Donchian upper band with volume confirmation and uptrend
            if close[i] > highest_high[i] and volume_confirmed and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian lower band with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and volume_confirmed and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to lower Donchian band or trend breaks
            if close[i] < lowest_low[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to upper Donchian band or trend breaks
            if close[i] > highest_high[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals