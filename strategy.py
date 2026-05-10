#!/usr/bin/env python3
# 4h_Donchian_Breakout_Trend_Volume_With_Stops
# Hypothesis: Donchian(20) breakouts with trend (1d EMA50) and volume confirmation work in both bull and bear markets.
# Uses 4h timeframe with 1d/1h multi-timeframe filters. Stops via signal=0 on breakdown below Donchian low.
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25.

name = "4h_Donchian_Breakout_Trend_Volume_With_Stops"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Get 1h data for volume confirmation (20-period average)
    df_1h = get_htf_data(prices, '1h')
    volume_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
    # Calculate Donchian channels (20-period) on 4h data
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_50_aligned[i]) or np.isnan(vol_ma_1h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 1h average
        volume_confirm = volume[i] > 1.5 * vol_ma_1h_aligned[i] if vol_ma_1h_aligned[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and above 1d EMA50 (uptrend)
            if close[i] > donchian_high[i] and volume_confirm and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and below 1d EMA50 (downtrend)
            elif close[i] < donchian_low[i] and volume_confirm and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: hold or exit
            if close[i] < donchian_low[i]:  # exit on breakdown below Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: hold or exit
            if close[i] > donchian_high[i]:  # exit on breakout above Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals