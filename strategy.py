#!/usr/bin/env python3
# 4h_Triple_Confirmation_Breakout_Strategy
# Hypothesis: Combine price channel breakout (Donchian 20), volume confirmation, and trend filter (12h EMA50) for high-probability entries.
# Uses discrete position sizing (0.25) to minimize churn. Designed for 4h timeframe to achieve 20-40 trades/year.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

name = "4h_Triple_Confirmation_Breakout_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) for breakout signals
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average volume
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 12h EMA50 to 4h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get 12h close for trend determination
        close_12h_series = pd.Series(close_12h)
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_series.values)
        
        is_uptrend = close_12h_aligned[i] > ema_50_12h_aligned[i]
        is_downtrend = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high in uptrend with volume
            if is_uptrend and close[i] > donchian_high[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low in downtrend with volume
            elif is_downtrend and close[i] < donchian_low[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian low or trend turns down
            if close[i] < donchian_low[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian high or trend turns up
            if close[i] > donchian_high[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals