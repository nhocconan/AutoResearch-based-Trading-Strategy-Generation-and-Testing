# [149849] 4h_Donchian_20_Breakout_1dTrend_Volume
# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# Works in bull markets (breakouts) and bear markets (breakdowns) by aligning with higher timeframe trend.
# Targets 20-50 trades/year on 4h timeframe with discrete sizing (0.25) to minimize fee churn.
# Volume confirmation reduces false breakouts; EMA filter ensures trading with higher timeframe momentum.

#!/usr/bin/env python3
name = "4h_Donchian_20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(30) for trend direction
    ema_30_1d = pd.Series(df_1d['close']).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 30)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_30_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA30
        price_above_ema = close[i] > ema_30_1d_aligned[i]
        price_below_ema = close[i] < ema_30_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 1d EMA30 + volume spike
            if (close[i] > high_20[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 1d EMA30 + volume spike
            elif (close[i] < low_20[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below Donchian low or volume drops below average
            if (close[i] < low_20[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above Donchian high or volume drops below average
            if (close[i] > high_20[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals