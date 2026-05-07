#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
# Long when Bull Power > 0 AND Bear Power < 0 AND EMA13 rising AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND Bull Power < 0 AND EMA13 falling AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period average.
# Exit when EMA13 flips direction or volume drops below average.
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 1d EMA34 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "6h_ElderRay_1dEMA34_VolumeFilter"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_rising[i]) or np.isnan(ema13_falling[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, EMA13 rising, price > 1d EMA34, volume filter
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and ema13_rising[i] and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0, EMA13 falling, price < 1d EMA34, volume filter
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and ema13_falling[i] and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA13 falling OR volume filter fails
            if ema13_falling[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA13 rising OR volume filter fails
            if ema13_rising[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals