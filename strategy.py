#!/usr/bin/env python3
"""
12h_donchian_breakout_volume_v2
Hypothesis: Donchian channel breakouts with volume confirmation and trend filter work in both bull and bear markets.
- Uses 12h timeframe for lower trade frequency (target: 20-40 trades/year)
- Entry: Price breaks above/below 20-period Donchian channel with volume > 1.5x average
- Trend filter: Only take longs when price > 50-period EMA, shorts when price < 50-period EMA
- Exit: Opposite Donchian breakout or trend reversal
- Volume filter reduces false breakouts, trend filter avoids counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

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
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_avg[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # EMA trend filter (50-period)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        
        if position == 1:  # Long
            # Exit: price breaks below lower Donchian or trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above upper Donchian or trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price breaks above upper Donchian with volume + uptrend
            if (close[i] > highest_high[i] and 
                vol_ratio > 1.5 and 
                close[i] > ema_50[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian with volume + downtrend
            elif (close[i] < lowest_low[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < ema_50[i]):
                position = -1
                signals[i] = -0.25
    
    return signals