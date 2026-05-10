#!/usr/bin/env python3
# 4h_Donchian_Breakout_Trend_Volume
# Hypothesis: In both bull and bear markets, breakouts from the 20-period Donchian channel
# with trend confirmation (4h EMA50) and volume spike lead to sustained moves.
# Uses 1d ADX as regime filter to avoid ranging markets. Designed for low trade frequency
# (20-40/year) to minimize fee drag while capturing strong trends.

name = "4h_Donchian_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and EMA calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channel (20-period)
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
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_4h[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0 and trending:
            # Long: price breaks above Donchian high with volume, above EMA50 (uptrend)
            if close[i] > donchian_high[i] and volume_confirm and close[i] > ema_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, below EMA50 (downtrend)
            elif close[i] < donchian_low[i] and volume_confirm and close[i] < ema_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low or below EMA50
            if close[i] < donchian_low[i] or close[i] < ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high or above EMA50
            if close[i] > donchian_high[i] or close[i] > ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals