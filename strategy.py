#!/usr/bin/env python3
# 4H_Donchian20_VolumeTrend_12hEMA50
# Hypothesis: 4h Donchian(20) breakout in direction of 12h EMA50 trend with volume confirmation.
# Works in bull/bear by following 12h trend. Entry on breakout with volume > 1.5x avg volume.
# Exit on opposite Donchian(10) breakout or trend reversal. Target: 20-40 trades/year.

name = "4H_Donchian20_VolumeTrend_12hEMA50"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Precompute 4h Donchian channels
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Exit channels (10-period)
    lookback_exit = 10
    upper_exit = np.full(n, np.nan)
    lower_exit = np.full(n, np.nan)
    for i in range(lookback_exit - 1, n):
        upper_exit[i] = np.max(high[i - lookback_exit + 1:i + 1])
        lower_exit[i] = np.min(low[i - lookback_exit + 1:i + 1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(49, n):  # start after warmup
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, uptrend, volume confirmation
            if close[i] > upper[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, downtrend, volume confirmation
            elif close[i] < lower[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower exit Donchian or trend turns down
            if close[i] < lower_exit[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper exit Donchian or trend turns up
            if close[i] > upper_exit[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals