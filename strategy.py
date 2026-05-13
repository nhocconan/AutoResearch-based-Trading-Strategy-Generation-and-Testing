#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and 1d volume confirmation.
# Long when price breaks above 6h Donchian upper (20) AND price > weekly EMA50 (bullish weekly trend) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 6h Donchian lower (20) AND price < weekly EMA50 (bearish weekly trend) AND 1d volume > 1.5x 20-period average.
# Exit when price crosses the 6h Donchian midpoint (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts aligned with weekly trend and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Donchian20_WeeklyEMA50_1dVolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 6h Donchian channels (20-period)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate Donchian channels for efficiency
    upper_20 = np.full(n, np.nan)
    lower_20 = np.full(n, np.nan)
    midpoint_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_20[i] = np.max(high[i-20:i])
        lower_20[i] = np.min(low[i-20:i])
        midpoint_20[i] = (upper_20[i] + lower_20[i]) / 2
    
    # Calculate weekly EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume average (20-period) for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(midpoint_20[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND price > weekly EMA50 AND volume > 1.5x average
            if (close[i] > upper_20[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > 1.5 * vol_ma_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND price < weekly EMA50 AND volume > 1.5x average
            elif (close[i] < lower_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > 1.5 * vol_ma_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses Donchian midpoint (mean reversion)
            if close[i] <= midpoint_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses Donchian midpoint (mean reversion)
            if close[i] >= midpoint_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals