#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike.
- Donchian(20) from prior 4h: price channel breakout structure
- Long: price breaks above upper band + volume > 1.8x 20-period avg + price > 1d EMA34
- Short: price breaks below lower band + volume > 1.8x 20-period avg + price < 1d EMA34
- Exit: price re-enters Donchian channel OR 1d EMA34 trend flip
- Uses Donchian for structure, volume for conviction, 1d EMA34 for HTF filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

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
    
    # Volume confirmation: > 1.8x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) from prior 4h OHLC (use lookback of 20 periods)
    # Upper band = max(high over past 20 periods)
    # Lower band = min(low over past 20 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band + volume confirmation + price > 1d EMA34
            if (close[i] > donchian_upper[i] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume confirmation + price < 1d EMA34
            elif (close[i] < donchian_lower[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below lower band (mean reversion) OR price < 1d EMA34 (trend flip)
            if close[i] < donchian_lower[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above upper band (mean reversion) OR price > 1d EMA34 (trend flip)
            if close[i] > donchian_upper[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0