#!/usr/bin/env python3
name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]  # Rising EMA
            
            if close[i] > high_20[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and 1d downtrend
            elif close[i] < low_20[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian midpoint or volume drops
            midpoint = (high_20[i] + low_20[i]) / 2
            if close[i] < midpoint or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian midpoint or volume drops
            midpoint = (high_20[i] + low_20[i]) / 2
            if close[i] > midpoint or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# - Donchian(20) breakouts capture momentum after range periods
# - 1d EMA(34) ensures alignment with higher timeframe trend
# - Volume spike (1.8x average) confirms institutional participation
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit at range midpoint provides logical profit target in ranging markets
# - Using 12h timeframe as requested in experiment #134985
# - 1d timeframe used for HTF trend filter as specified in experiment instructions