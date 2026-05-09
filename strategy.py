#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (bull/bear power) with 1d trend filter and volume confirmation
# Long when Bull Power > 0, Bear Power < 0, 1d EMA50 > EMA200 (bullish trend), and volume > 1.5x average
# Short when Bull Power < 0, Bear Power > 0, 1d EMA50 < EMA200 (bearish trend), and volume > 1.5x average
# Exit when Elder Ray signals reverse or trend weakens
# Combines momentum (Elder Ray) with trend (EMA crossover) and volume to capture sustained moves
# Designed for medium-frequency, high-conviction trades on 6h timeframe
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_ElderRay_Trend_Volume"
timeframe = "6h"
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
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Trend: 1 if EMA50 > EMA200 (bullish), -1 if EMA50 < EMA200 (bearish)
    trend_1d = np.where(ema50_1d > ema200_1d, 1, -1)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, bullish trend, volume confirmation
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and
                trend_aligned[i] == 1 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power < 0, Bear Power > 0, bearish trend, volume confirmation
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and
                  trend_aligned[i] == -1 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Elder Ray turns bearish or trend turns bearish
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or
                trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Elder Ray turns bullish or trend turns bullish
            if (bull_power[i] >= 0 or 
                bear_power[i] <= 0 or
                trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals