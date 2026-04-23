#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + ADX regime filter
Donchian channel breakouts capture sustained momentum. Volume confirmation ensures
institutional participation. ADX > 25 filters for trending markets, avoiding whipsaws
in sideways action. Works in both bull (breakouts up) and bear (breakdowns down).
Target: 20-40 trades/year to minimize fee drag.
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
    
    # Calculate ADX(14) for regime filter
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    plus_dm_smooth = wilder_smooth(plus_dm, period_adx)
    minus_dm_smooth = wilder_smooth(minus_dm, period_adx)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period_adx)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average (tighter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # need Donchian, vol MA, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high + volume spike + ADX > 25 (trending)
            if (close[i] > donchian_high[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low + volume spike + ADX > 25 (trending)
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR ADX < 20 (trend weakening)
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian low OR ADX < 20
                if close[i] < donchian_low[i] or adx[i] < 20:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian high OR ADX < 20
                if close[i] > donchian_high[i] or adx[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0