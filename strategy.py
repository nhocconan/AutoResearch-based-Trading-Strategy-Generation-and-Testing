#!/usr/bin/env python3
"""
6h_ADX_Trend_With_VolumeSpike
Hypothesis: Use ADX(14) to confirm trend strength (>25) on 6h, enter long when price breaks above 6h Donchian(20) high with volume > 2x 20-period average, short when breaks below Donchian low with volume > 2x average. Exit when price crosses the 6h EMA(34) in the opposite direction. Uses 1d ADX as trend filter to ensure alignment with higher timeframe trend. Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years). Works in bull markets by following strong trends and in bear markets by avoiding counter-trend trades via higher timeframe ADX filter.
"""

name = "6h_ADX_Trend_With_VolumeSpike"
timeframe = "6h"
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
    
    # Calculate 6h ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros(len(high))
        adx = np.zeros(len(high))
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        adx[2*period-1] = np.mean(dx[2*period-1:3*period])
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Calculate 1d ADX for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Calculate 6h EMA(34) for exit
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False).mean().values
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(adx_6h[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Require both 6h and 1d ADX > 25 for trend strength
        if adx_6h[i] <= 25 or adx_1d_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike
            if close[i] > donch_high[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike
            elif close[i] < donch_low[i] and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: Exit when price crosses below EMA(34)
            if close[i] < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: Exit when price crosses above EMA(34)
            if close[i] > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals