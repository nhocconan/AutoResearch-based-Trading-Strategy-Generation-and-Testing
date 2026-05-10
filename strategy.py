#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: Donchian(20) breakouts with volume confirmation and trend filter (EMA25) capture institutional breakouts in both bull and bear markets.
# Uses volume > 1.5x 20-period average to filter false breakouts. Trend filter ensures alignment with 4h EMA25 direction.
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe.

name = "4h_Donchian_Breakout_Volume_Trend"
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
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Trend filter: EMA25 on 4h close
    ema25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    trend_up = close > ema25
    trend_down = close < ema25
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Need enough data for EMA25
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend
            if (high[i] > highest_high[i] and
                trend_up[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and downtrend
            elif (low[i] < lowest_low[i] and
                  trend_down[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low or trend turns down
            if (low[i] < lowest_low[i] or
                not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high or trend turns up
            if (high[i] > highest_high[i] or
                not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals