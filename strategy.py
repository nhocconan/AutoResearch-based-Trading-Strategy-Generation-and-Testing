#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, 1d EMA34 rising, and 6h volume > 1.5x 20-period average.
# Short when Bull Power < 0, Bear Power > 0, 1d EMA34 falling, and 6h volume > 1.5x 20-period average.
# Exit when Elder Ray signals reverse or volume drops.
# Elder Ray captures bull/bear power via EMA13 relative to high/low, effective in trending markets.
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
    
    # 6h Elder Ray components (EMA13-based)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
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
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Determine 1d EMA34 slope (rising/falling) - compare to previous value
    ema34_rising = np.zeros_like(ema34_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_aligned, dtype=bool)
    ema34_rising[1:] = ema34_aligned[1:] > ema34_aligned[:-1]
    ema34_falling[1:] = ema34_aligned[1:] < ema34_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, 1d EMA34 rising, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and ema34_rising[i] and volume_filter[i]
            # Short conditions: Bull Power < 0, Bear Power > 0, 1d EMA34 falling, volume spike
            short_cond = (bull_power[i] < 0) and (bear_power[i] > 0) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or Bear Power >= 0 or volume drops
            if (bull_power[i] <= 0) or (bear_power[i] >= 0) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power >= 0 or Bear Power <= 0 or volume drops
            if (bull_power[i] >= 0) or (bear_power[i] <= 0) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals