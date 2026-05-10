#!/usr/bin/env python3
# 12H_Donchian_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Uses Donchian channel breakout on 12h chart filtered by 1-day trend (close > EMA100) and volume spike.
# Enters long when price breaks above Donchian upper channel with volume > 1.5x average volume and in uptrend.
# Enters short when price breaks below Donchian lower channel with volume > 1.5x average volume and in downtrend.
# Exits when price crosses the Donchian middle channel or volume drops below average.
# Uses 1-day EMA100 for trend filter to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25.

name = "12H_Donchian_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend (EMA100)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA(100) for trend direction
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate Donchian channel on 12h
    period = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    middle = np.full_like(high, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Calculate average volume for confirmation
    vol_ma = np.full_like(volume, np.nan)
    for i in range(period-1, len(volume)):
        vol_ma[i] = np.mean(volume[i-period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 100)  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_100_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA100
        price_above_ema = close[i] > ema_100_1d_aligned[i]
        price_below_ema = close[i] < ema_100_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with volume confirmation in uptrend
            if (close[i] > upper[i] and 
                volume_confirmed and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume confirmation in downtrend
            elif (close[i] < lower[i] and 
                  volume_confirmed and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below middle Donchian or volume drops below average
            if (close[i] < middle[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above middle Donchian or volume drops below average
            if (close[i] > middle[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals