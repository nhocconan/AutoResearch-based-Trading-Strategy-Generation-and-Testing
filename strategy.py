#!/usr/bin/env python3
name = "12h_Weekly_Donchian_Breakout_Trend_Volume"
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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 12h Donchian channel (20 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high in weekly uptrend with volume
            if close[i] > donchian_high[i] and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in weekly downtrend with volume
            elif close[i] < donchian_low[i] and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and volume[i] > vol_ma_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or trend change
            if close[i] < donchian_low[i] or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or trend change
            if close[i] > donchian_high[i] or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation
# - Uses weekly EMA20 trend to ensure alignment with higher timeframe direction
# - Donchian breakout (20-period) provides clear entry/exit levels
# - Volume confirmation (1.5x average) reduces false breakouts
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Position size 0.25 targets ~15-35 trades/year to stay within limits
# - Simple, robust logic with minimal overfitting risk
# - Weekly trend filter reduces whipsaws vs same-timeframe signals
# - Designed for low trade frequency to minimize fee drag (critical for 12h)