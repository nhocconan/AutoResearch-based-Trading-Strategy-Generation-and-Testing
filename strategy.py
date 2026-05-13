#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike confirmation and 1w ADX regime filter.
# Long when price breaks above 20-period 12h Donchian high AND 1d volume > 1.5x 20-period average AND 1w ADX > 25 (trending market).
# Short when price breaks below 20-period 12h Donchian low AND 1d volume > 1.5x 20-period average AND 1w ADX > 25.
# Exit when price returns to 12-period 12h Donchian midpoint OR 1w ADX < 20 (regime shift to ranging).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts in trending markets while avoiding false signals in ranging markets via ADX filter.

name = "12h_DonchianBreakout_VolumeSpike_ADXRegime_v1"
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
    
    # Calculate 12h Donchian(20) channels (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian high and low (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_12h, 20)
    donchian_low = rolling_min(low_12h, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Calculate 1d volume spike confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    avg_volume_20 = rolling_mean(volume_1d, 20)
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1w ADX for regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1w))
    minus_dm = np.zeros(len(low_1w))
    tr = np.zeros(len(high_1w))
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(0, high_1w[i] - high_1w[i-1])
        minus_dm[i] = max(0, low_1w[i-1] - low_1w[i])
        tr[i] = max(high_1w[i] - low_1w[i], 
                   abs(high_1w[i] - close_1w[i-1]), 
                   abs(low_1w[i] - close_1w[i-1]))
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])  # first value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) < period:
        return np.zeros(n)
        
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    divisor = np.where(atr == 0, 1, atr)
    plus_di = 100 * plus_dm_smooth / divisor
    minus_di = 100 * minus_dm_smooth / divisor
    
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = wilder_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after sufficient lookback
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND ADX > 25 (trending)
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and  # volume spike confirmed
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND ADX > 25 (trending)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and  # volume spike confirmed
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian midpoint OR ADX < 20 (regime shift to ranging)
            if (close[i] <= donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian midpoint OR ADX < 20 (regime shift to ranging)
            if (close[i] >= donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals