#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
# Donchian breakouts capture strong momentum moves. 12h EMA50 ensures alignment with higher timeframe trend to avoid false breakouts.
# Volume spike (2x 20-period volume EMA) confirms breakout validity. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in bull markets via upside breakouts in uptrend and bear markets via downside breakouts in downtrend.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 12h uptrend AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_aligned[i] and  # 12h uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 12h downtrend AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_aligned[i] and  # 12h downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR 12h trend turns down
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] < donchian_middle or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian middle OR 12h trend turns up
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] > donchian_middle or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals