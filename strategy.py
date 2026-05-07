#!/usr/bin/env python3
name = "4h_Donchian_Breakout_20_12hTrend_Volume"
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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above upper Donchian in 12h uptrend with volume
            if close[i] > high_20[i] and ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian in 12h downtrend with volume
            elif close[i] < low_20[i] and ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below lower Donchian
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above upper Donchian
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 4h with 12h EMA trend filter and volume confirmation
# - Long when price breaks above 20-period high + 12h EMA rising + volume spike
# - Short when price breaks below 20-period low + 12h EMA falling + volume spike
# - Exit when price returns to opposite Donchian band
# - Volume confirmation (2x average) reduces false breakouts
# - Position size 0.25 limits drawdown and reduces trade frequency
# - Target: 20-50 trades/year to stay within fee drag limits
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# - Uses 12h trend filter to avoid counter-trend trades
# - Simple, robust structure with clear entry/exit rules