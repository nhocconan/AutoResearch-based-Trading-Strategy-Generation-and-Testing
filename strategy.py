#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_1dTrend_Filter
# Hypothesis: Price breaks above/below 20-period Donchian channel on 12h timeframe,
#             confirmed by 1d EMA trend and volume spike. Works in bull markets via
#             breakouts above channel and in bear via breakdowns below channel.
#             Target: 15-25 trades/year on 12h timeframe (60-100 total over 4 years).

name = "12h_PriceChannel_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily timeframe (trend filter)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Donchian channel and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channel on 12h
    period = 20
    high_max = pd.Series(high).rolling(window=period, min_periods=period).max()
    low_min = pd.Series(low).rolling(window=period, min_periods=period).min()
    upper_channel = high_max.values
    lower_channel = low_min.values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + EMA (50) + vol EMA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND above 1d EMA50 AND volume spike
            if close[i] > upper_channel[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND below 1d EMA50 AND volume spike
            elif close[i] < lower_channel[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below lower channel
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper channel
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals