#!/usr/bin/env python3
name = "4h_Donchian_Breakout_20_1dTrend_Volume"
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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian in daily uptrend with volume
            if close[i] > high_20[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian in daily downtrend with volume
            elif close[i] < low_20[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume[i] > vol_ma_20[i] * 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below lower Donchian or trend change
            if close[i] < low_20[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above upper Donchian or trend change
            if close[i] > high_20[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# - Long when price breaks above 20-period high with daily uptrend and volume spike
# - Short when price breaks below 20-period low with daily downtrend and volume spike
# - Exit when price returns to opposite Donchian band or trend changes
# - Volume filter (2x average) reduces false breakouts
# - Position size 0.25 limits drawdown and keeps trade count manageable
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Target: ~30-50 trades/year to avoid fee drag while capturing trends
# - Proven pattern: Donchian breakouts + trend + volume works on SOL/ETH/USDT pairs