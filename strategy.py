# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Go long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA34 > prior 1d EMA34 (uptrend) AND volume > 1.5x 20-period average.
# Go short when Bear Power > 0 AND Bull Power < 0 AND 1d EMA34 < prior 1d EMA34 (downtrend) AND volume > 1.5x 20-period average.
# Exit when power signals weaken or reverse.
# Uses Elder Ray to measure bull/bear strength relative to trend EMA, filtered by higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "6h_ElderRay_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components on 6h
    ema13_period = 13
    ema13 = pd.Series(close).ewm(span=ema13_period, adjust=False, min_periods=ema13_period).mean().values
    bull_power = high - ema13  # High minus EMA
    bear_power = ema13 - low   # EMA minus Low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Trend: current EMA34 > previous EMA34 (up), or < (down)
    ema34_up = ema34_1d > np.roll(ema34_1d, 1)
    ema34_down = ema34_1d < np.roll(ema34_1d, 1)
    ema34_up[0] = False  # First value has no previous
    ema34_down[0] = False
    
    # Align 1d EMA trend signals to 6h timeframe
    ema34_up_aligned = align_htf_to_ltf(prices, df_1d, ema34_up.astype(float))
    ema34_down_aligned = align_htf_to_ltf(prices, df_1d, ema34_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_up_aligned[i]) or 
            np.isnan(ema34_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power positive, Bear Power negative, 1d uptrend, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (ema34_up_aligned[i] > 0.5) and volume_filter[i]
            # Short conditions: Bear Power positive, Bull Power negative, 1d downtrend, volume spike
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and (ema34_down_aligned[i] > 0.5) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Bear Power turns positive (weakening bulls)
            if (bull_power[i] <= 0) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative OR Bull Power turns positive (weakening bears)
            if (bear_power[i] <= 0) or (bull_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals