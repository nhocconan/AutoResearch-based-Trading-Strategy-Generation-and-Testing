#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND 1d EMA34 > prior 1d EMA34 (uptrend) AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power < 0 AND 1d EMA34 < prior 1d EMA34 (downtrend) AND volume > 1.5x 20-period average.
# Exit when Bull Power and Bear Power both turn negative (for long) or both turn positive (for short).
# Elder Ray measures bull/bear strength relative to EMA13, effective in both bull and bear markets when combined with trend filter.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_ElderRay_1dEMA34_Volume"
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
    
    # 6h EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d EMA34 trend: rising if current > previous, falling if current < previous
    ema34_rising = ema34_1d > np.roll(ema34_1d, 1)
    ema34_falling = ema34_1d < np.roll(ema34_1d, 1)
    ema34_rising[0] = False
    ema34_falling[0] = False
    
    # Align 1d EMA34 trend to 6h timeframe
    ema34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema34_rising)
    ema34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema34_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_rising_aligned[i]) or 
            np.isnan(ema34_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, 1d EMA34 rising, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and ema34_rising_aligned[i] and volume_filter[i]
            # Short conditions: Bear Power < 0, Bull Power < 0, 1d EMA34 falling, volume spike
            short_cond = (bear_power[i] < 0) and (bull_power[i] < 0) and ema34_falling_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 (loss of bullish bias)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power >= 0 OR Bear Power <= 0 (loss of bearish bias)
            if bull_power[i] >= 0 or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals