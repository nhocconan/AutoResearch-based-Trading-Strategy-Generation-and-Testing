#!/usr/bin/env python3
name = "1d_4H_Trend_Align_Breakout"
timeframe = "1d"
leverage = 1.0

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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above 20-day high in 4h uptrend with volume
            if close[i] > high_max_20[i] and ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low in 4h downtrend with volume
            elif close[i] < low_min_20[i] and ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close back below 20-day average or trend reversal
            mid_point = (high_max_20[i] + low_min_20[i]) / 2
            if close[i] < mid_point or ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above 20-day average or trend reversal
            mid_point = (high_max_20[i] + low_min_20[i]) / 2
            if close[i] > mid_point or ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakouts aligned with 4h EMA trend and volume confirmation
# - Uses 20-day Donchian channels for breakout detection
# - 4h EMA(34) trend filter ensures trades align with intermediate trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exits when price returns to midpoint of channel or trend reverses
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Position size 0.25 targets ~40-80 trades/year to avoid fee drag
# - Simple, robust logic with clear entry/exit conditions
# - Aims for 50-100 total trades over 4 years (12-25/year) to stay within limits
# - Focus on BTC/ETH as primary targets; avoids over-optimization on SOL
# - Weekly timeframe not needed as 4h provides sufficient trend context for daily signals