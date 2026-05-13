#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: Donchian(20) breakout on 12h with 1d EMA50 trend filter and volume spike.
# Uses 12h timeframe for low frequency, 1d for trend confirmation. Designed to work in both bull and bear markets via trend filter.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 20-period average (high threshold for low frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    # Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Breakout conditions
        price_above_donchian_high = close[i] > donchian_high[i]
        price_below_donchian_low = close[i] < donchian_low[i]
        
        # Trend conditions
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above Donchian high + uptrend + volume spike
            if price_above_donchian_high and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + downtrend + volume spike
            elif price_below_donchian_low and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR trend reversal
            if price_below_donchian_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR trend reversal
            if price_above_donchian_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals