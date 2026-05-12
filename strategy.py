#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_VolumeFilter
# Hypothesis: Use Donchian channel breakout with daily EMA trend filter and volume spike.
# Long when price breaks above upper Donchian band with price > daily EMA and volume > 2x MA.
# Short when price breaks below lower Donchian band with price < daily EMA and volume > 2x MA.
# Exit when price reverses back into the Donchian channel.
# Designed to capture strong trending moves with confirmation, works in both bull and bear markets by filtering with daily trend.
# Targets 15-30 trades/year to minimize fee drag.

name = "12h_Donchian_Breakout_1dTrend_VolumeFilter"
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
    
    # Calculate Donchian Channel (20-period)
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band with price > daily EMA and volume > 2x MA
            if close[i] > upper[i] and close[i] > daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with price < daily EMA and volume > 2x MA
            elif close[i] < lower[i] and close[i] < daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below upper band
            if close[i] < upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above lower band
            if close[i] > lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals