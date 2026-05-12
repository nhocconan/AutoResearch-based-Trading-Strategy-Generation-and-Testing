#!/usr/bin/env python3
# 4h_Donchian_20_Volume_Spike_1d_Trend_HTF
# Hypothesis: 4-hour Donchian breakout with volume confirmation and 1-day trend filter.
# Works in bull markets: breakouts above 20-period high with volume and uptrend.
# Works in bear markets: breakdowns below 20-period low with volume and downtrend.
# Uses 1-day EMA50 for trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian_20_Volume_Spike_1d_Trend_HTF"
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
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50 and Donchian
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + volume spike + above 1d EMA50 (uptrend)
            if (close[i] > high_20[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + volume spike + below 1d EMA50 (downtrend)
            elif (close[i] < low_20[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters the 20-period range OR closes below 1d EMA50
            if (close[i] < high_20[i] and close[i] > low_20[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters the 20-period range OR closes above 1d EMA50
            if (close[i] < high_20[i] and close[i] > low_20[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals