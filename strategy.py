#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_Filter
Hypothesis: On 4h timeframe, price breaking Donchian(20) channels with volume confirmation 
and EMA50 trend filter captures sustained moves. Works in bull markets (breakouts to upside) 
and bear markets (breakdowns to downside). Volume ensures breakout legitimacy, EMA50 
filters counter-trend noise. Target 20-50 trades/year to minimize fee drag.
"""

name = "4h_Donchian20_Breakout_VolumeTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) - highest high and lowest low
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # EMA50 for trend filter on 4h data
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema50[49] = np.mean(close[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, n):
            ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 50)  # Need Donchian and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema50[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and uptrend
            if close[i] > highest_high[i] and volume_confirm and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and downtrend
            elif close[i] < lowest_low[i] and volume_confirm and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls below Donchian low or trend reversal
            if close[i] < lowest_low[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises above Donchian high or trend reversal
            if close[i] > highest_high[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals