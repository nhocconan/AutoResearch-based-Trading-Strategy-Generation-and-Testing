#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Long when Bull Power > 0 AND Bear Power < 0 AND price > EMA13 (1d) AND volume > 1.5x 20-period average
# Short when Bear Power > 0 AND Bull Power < 0 AND price < EMA13 (1d) AND volume > 1.5x 20-period average
# Exit when power signals reverse or volume drops
# Elder Ray measures bull/bear strength via EMA13; 1d EMA13 filters regime; volume confirms strength
# Target: 50-150 total trades over 4 years (12-37/year) on 6h

name = "6h_ElderRay_1dEMA13_Volume_v1"
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
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d for regime
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d EMA13 to 6h timeframe (wait for 1d bar to close)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient data for EMA13
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA13 AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema13_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA13 AND volume confirmation
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema13_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR Bear Power >= 0 OR volume drops
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 OR Bull Power >= 0 OR volume drops
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals