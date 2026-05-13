#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1D for trend filter
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1D EMA13 to 6H timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Use 6H EMA13 for consistency with Elder Ray calculation
    close_series = pd.Series(close)
    ema13_6h = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema13_6h[i])):
            signals[i] = 0.0
            continue
        
        # 1D trend filter: only trade in direction of 1D EMA13 trend
        # Bullish trend: price above EMA13, Bearish trend: price below EMA13
        bullish_trend = close[i] > ema13_1d_aligned[i]
        bearish_trend = close[i] < ema13_1d_aligned[i]
        
        if position == 0:
            # LONG: Bullish 1D trend + Bull Power > 0 (bulls in control)
            if bullish_trend and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 1D trend + Bear Power < 0 (bears in control)
            elif bearish_trend and bear_power[i] < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish 1D trend OR Bull Power turns negative
            if (not bullish_trend) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish 1D trend OR Bear Power turns positive
            if (not bearish_trend) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals