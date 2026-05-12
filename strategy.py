#!/usr/bin/env python3
# 12h_Donchian_Breakout_1wTrend_VolumeFilter
# Hypothesis: On 12h timeframe, enter long when price breaks above 20-period Donchian high with weekly EMA50 trend alignment and volume confirmation.
# Enter short when price breaks below 20-period Donchian low with weekly EMA50 trend alignment and volume confirmation.
# Exit when price crosses weekly EMA50 (trend reversal).
# Uses weekly trend filter to reduce whipsaw and improve performance in both bull and bear markets.
# Targets 15-25 trades/year for low fee drag.

name = "12h_Donchian_Breakout_1wTrend_VolumeFilter"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        weekly_trend = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high with weekly trend up and volume confirmation
            if close[i] > donchian_high_val and close[i] > weekly_trend and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with weekly trend down and volume confirmation
            elif close[i] < donchian_low_val and close[i] < weekly_trend and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA50 (trend reversal)
            if close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA50 (trend reversal)
            if close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals