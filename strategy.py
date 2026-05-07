#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bull/Bear Power (Elder Ray) with weekly trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 AND weekly EMA50 rising AND price > weekly EMA50 AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND Bull Power < 0 AND weekly EMA50 falling AND price < weekly EMA50 AND volume > 1.5x 20-period average.
# Exit when weekly EMA50 flips direction or volume drops below average.
# Uses weekly EMA50 for trend filter to avoid counter-trend trades in strong trends and capture major trend moves.
# Volume filter ensures participation and avoids low-conviction moves.
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
name = "6h_ElderRay_WeeklyEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray and trend
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # EMA13 direction
    ema13_rising = np.zeros_like(ema13, dtype=bool)
    ema13_falling = np.zeros_like(ema13, dtype=bool)
    ema13_rising[1:] = ema13[1:] > ema13[:-1]
    ema13_falling[1:] = ema13[1:] < ema13[:-1]
    
    # Weekly EMA50 for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly EMA50 direction
    ema50_rising = np.zeros_like(ema50_weekly_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_weekly_aligned, dtype=bool)
    ema50_rising[1:] = ema50_weekly_aligned[1:] > ema50_weekly_aligned[:-1]
    ema50_falling[1:] = ema50_weekly_aligned[1:] < ema50_weekly_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_rising[i]) or np.isnan(ema13_falling[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, weekly EMA50 rising, price > weekly EMA50, volume filter
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and ema50_rising[i] and (close[i] > ema50_weekly_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0, weekly EMA50 falling, price < weekly EMA50, volume filter
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and ema50_falling[i] and (close[i] < ema50_weekly_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly EMA50 falling OR volume filter fails
            if ema50_falling[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly EMA50 rising OR volume filter fails
            if ema50_rising[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals