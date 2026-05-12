#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: On 4h timeframe, use Donchian channel breakouts (20-period) as primary entry signals.
# Filter with 1d EMA50 trend and volume confirmation (current volume > 1.5x 20-period average).
# Exit on opposite Donchian band touch or when trend reverses.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 20-50 trades/year to minimize fee drag while capturing significant moves.

name = "4h_Donchian_Breakout_Volume_Trend"
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(daily_ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        daily_trend = daily_ema50_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band with volume confirmation and daily uptrend
            if close[i] > upper_channel and close[i] > daily_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band with volume confirmation and daily downtrend
            elif close[i] < lower_channel and close[i] < daily_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches lower Donchian band or trend turns down
            if close[i] < lower_channel or close[i] < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches upper Donchian band or trend turns up
            if close[i] > upper_channel or close[i] > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals