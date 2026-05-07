#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) on 12h price
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 8-period average (4 days of 12h bars)
    vol_ma_8 = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 8)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_8[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_8[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > high_20[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and daily downtrend
            elif close[i] < low_20[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_8[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_8[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with daily trend and volume confirmation
# - Breakout above 20-period high with volume in daily uptrend = long opportunity
# - Breakdown below 20-period low with volume in daily downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~15-25 trades/year, avoiding fee drag
# - Daily EMA(34) filter ensures alignment with higher timeframe trend
# - Donchian channels provide robust structure that adapts to volatility
# - Simple 3-condition entry reduces overtrading and improves generalization