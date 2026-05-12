#!/usr/bin/env python3
# 4h_Keltner_Breakout_1dTrend_VolumeFilter
# Hypothesis: On 4h timeframe, enter long when price breaks above Keltner upper band with price > daily EMA34 and volume > 1.5x average.
# Enter short when price breaks below Keltner lower band with price < daily EMA34 and volume > 1.5x average.
# Exit when price crosses back inside the Keltner bands.
# Uses daily timeframe for trend filter to avoid counter-trend trades.
# Targets 20-40 trades/year for low fee drag and works in both bull and bear markets by following trends with volatility-adjusted bands.

name = "4h_Keltner_Breakout_1dTrend_VolumeFilter"
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
    
    # Calculate Keltner Channel from 4h data
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.abs(high - low)).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume confirmation: 20-period moving average on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema20[i]) or np.isnan(atr[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        daily_trend = daily_ema34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above upper band with price > daily EMA34 and volume > 1.5x MA
            if close[i] > upper and close[i] > daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with price < daily EMA34 and volume > 1.5x MA
            elif close[i] < lower and close[i] < daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside Keltner bands (below upper)
            if close[i] < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside Keltner bands (above lower)
            if close[i] > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals