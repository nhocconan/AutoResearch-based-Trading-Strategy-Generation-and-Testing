#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Uses Donchian channels from 4h chart to identify key breakout levels.
# Enters long when price breaks above upper Donchian(20) with volume confirmation and ATR trend filter.
# Enters short when price breaks below lower Donchian(20) with volume confirmation and ATR trend filter.
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Donchian provides structure, volume confirms breakout validity, ATR filter ensures trending conditions.
# Works in bull markets via breakouts and in bear markets via breakdowns with proper filtering.

name = "4h_Donchian20_VolumeSpike_ATRTrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) - using only past data
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for trend filter and volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR-based trend filter: price > EMA(close, 50) + 0.5*ATR for uptrend
    # price < EMA(close, 50) - 0.5*ATR for downtrend
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_trend_up = ema50 + 0.5 * atr
    ema_trend_down = ema50 - 0.5 * atr
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(ema_trend_up[i]) or np.isnan(ema_trend_down[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND volume spike AND uptrend filter
            if (close[i] > high_rolling_max[i] and 
                volume_spike[i] and 
                close[i] > ema_trend_up[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND volume spike AND downtrend filter
            elif (close[i] < low_rolling_min[i] and 
                  volume_spike[i] and 
                  close[i] < ema_trend_down[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR trend reverses
            if (close[i] >= low_rolling_min[i] and close[i] <= high_rolling_max[i]) or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR trend reverses
            if (close[i] >= low_rolling_min[i] and close[i] <= high_rolling_max[i]) or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals