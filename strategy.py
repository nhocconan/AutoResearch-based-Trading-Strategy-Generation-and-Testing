#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_VolumeSpike_TrendFilter
Hypothesis: Donchian(20) breakouts on 4h capture medium-term trends. Volume confirmation (>2x 24-period average) filters false breakouts. Trend filter uses 4h EMA50 to ensure alignment with intermediate trend. Designed for low trade frequency (~20-40/year) to minimize fee drag in 4-hour bars. Works in both bull and bear markets by using trend filter to avoid counter-trend entries.
"""

name = "4h_Donchian_Breakout_20_VolumeSpike_TrendFilter"
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
    
    # Donchian channel (20-period) - calculated on price data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation: current volume > 2.0x 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Trend filter: EMA(50) on 4h close
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above Donchian upper, volume confirmation, price above EMA50 (uptrend)
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, volume confirmation, price below EMA50 (downtrend)
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian upper (failed breakout) OR volume drops
            if (close[i] < donchian_upper[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian lower (failed breakdown) OR volume drops
            if (close[i] > donchian_lower[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals