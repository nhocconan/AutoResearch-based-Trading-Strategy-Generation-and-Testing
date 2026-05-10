#!/usr/bin/env python3
# 1H_4h1D_Trend_Momentum_With_Volume
# Hypothesis: In both bull and bear markets, momentum aligned with 4h trend and confirmed by volume and 1d structure leads to sustainable moves.
# Uses 4h EMA20 for trend direction, 1d high/low for structure, and volume spike for confirmation.
# Designed for low trade frequency (15-30/year) to minimize fee drag on 1h timeframe.

name = "1H_4h1D_Trend_Momentum_With_Volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend (responsive but smooth)
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for structure (prior day's high-low)
    df_1d = get_htf_data(prices, '1d')
    # Prior day's high and low (use previous day's values)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    # Align daily levels to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation (24-period average on 1h = 1 day)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above prior day's high with volume, above 4h EMA20 (uptrend)
            if close[i] > prev_high_aligned[i] and volume_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below prior day's low with volume, below 4h EMA20 (downtrend)
            elif close[i] < prev_low_aligned[i] and volume_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below prior day's low or breaks below 4h EMA20
            if close[i] < prev_low_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above prior day's high or breaks above 4h EMA20
            if close[i] > prev_high_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals