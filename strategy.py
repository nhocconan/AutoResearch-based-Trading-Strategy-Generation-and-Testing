#!/usr/bin/env python3
# 4h_Multi_Filter_Breakout_Strategy
# Hypothesis: Combine Donchian breakout with 12h EMA trend filter, volume confirmation, and ADX regime filter.
# Designed to work in both bull and bear markets by filtering trades with trend and momentum.
# Target: 25-40 trades/year to avoid fee drag while maintaining edge.

name = "4h_Multi_Filter_Breakout_Strategy"
timeframe = "4h"
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
    
    # Donchian Channel (20-period)
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
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average volume
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    
    vol_ma_20 = mean_arr(volume, 20)
    
    # ADX (14-period) for regime filter
    def rma(arr, length):
        res = np.full_like(arr, np.nan)
        if len(arr) < length:
            return res
        alpha = 1.0 / length
        res[length - 1] = np.mean(arr[:length])
        for i in range(length, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
        return res
    
    def adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        atr = rma(tr, length)
        plus_di = 100 * rma(plus_dm, length) / (atr + 1e-10)
        minus_di = 100 * rma(minus_dm, length) / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return rma(dx, length)
    
    adx_vals = adx(high, low, close, 14)
    adx_filter = adx_vals > 20  # Only trade when ADX > 20 (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or \
           np.isnan(adx_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high in uptrend with volume and ADX filter
            if is_uptrend and close[i] > donchian_high[i] and volume_condition and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low in downtrend with volume and ADX filter
            elif is_downtrend and close[i] < donchian_low[i] and volume_condition and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian low or trend turns down
            if close[i] < donchian_low[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian high or trend turns up
            if close[i] > donchian_high[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals