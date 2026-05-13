#!/usr/bin/env python3
# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Uses 4h EMA50 as trend direction, enters on 1h Donchian breakout in trend direction when volume is above average.
# Designed for low trade frequency (~15-30/year) by requiring 4h trend alignment and volume confirmation.
# Donchian channels provide clear breakout levels, and 4h trend filter avoids counter-trend trades.

name = "1h_Donchian_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if not session_ok[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema50_4h_aligned[i] and volume_ok[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Donchian low with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema50_4h_aligned[i] and volume_ok[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low or trend changes
            if close[i] < donchian_low[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high or trend changes
            if close[i] > donchian_high[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals