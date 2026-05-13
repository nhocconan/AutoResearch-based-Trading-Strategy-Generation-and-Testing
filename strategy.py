#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-bar Donchian high AND price > 1w EMA50 AND 1d volume > 1.5 * 20-bar average volume.
# Short when price breaks below 20-bar Donchian low AND price < 1w EMA50 AND 1d volume > 1.5 * 20-bar average volume.
# Exit when price crosses the 10-bar Donchian midpoint (mean reversion) or adverse 1w EMA50 crossover.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing
# medium-term breakouts with institutional volume confirmation while avoiding false signals in low-volume environments.

name = "12h_Donchian20_EMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume spike confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate Donchian channels (20-bar) and midpoint (10-bar) on primary timeframe
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high_20 = rolling_max(high, 20)
    donchian_low_20 = rolling_min(low, 20)
    donchian_mid_10 = (rolling_max(high, 10) + rolling_min(low, 10)) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-bar Donchian high AND above 1w EMA50 AND volume spike
            if (close[i] > donchian_high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike_aligned[i] > 0.5):  # boolean as 0/1
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-bar Donchian low AND below 1w EMA50 AND volume spike
            elif (close[i] < donchian_low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 10-bar Donchian midpoint OR below 1w EMA50
            if (close[i] < donchian_mid_10[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 10-bar Donchian midpoint OR above 1w EMA50
            if (close[i] > donchian_mid_10[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals