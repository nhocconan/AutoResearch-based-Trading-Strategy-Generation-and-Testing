#!/usr/bin/env python3
# 12h_donchian_20_volume_regime_v1
# Hypothesis: Donchian(20) breakout on 12h with volume confirmation (>2x average) and trend filter (ADX>20) captures breakouts in both bull and bear markets. Uses 1d for trend context and ATR-based stoploss. Target: 12-37 trades/year (50-150 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    for i in range(14, n):
        plus_dm_sum = np.sum(plus_dm[i-13:i+1])
        minus_dm_sum = np.sum(minus_dm[i-13:i+1])
        tr_sum = np.sum(tr[i-13:i+1])
        if tr_sum > 0:
            plus_di[i] = 100 * plus_dm_sum / tr_sum
            minus_di[i] = 100 * minus_dm_sum / tr_sum
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(n)
    adx[13] = dx[13]
    for i in range(14, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Get daily trend context (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros(len(df_1d))
    ema_50_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i-19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_12h[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma_20[i] * 2.0
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_12h[i]
        downtrend = close[i] < ema_50_12h[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if close[i] < donchian_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if close[i] > donchian_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and uptrend
            if close[i] > donchian_high[i] and vol_spike and uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and downtrend
            elif close[i] < donchian_low[i] and vol_spike and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals