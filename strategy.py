#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with EMA200 trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum), price > EMA200 (uptrend), volume > 1.5x 20-period average.
# Short when Bear Power > 0 and Bull Power < 0 (bearish momentum), price < EMA200 (downtrend), volume > 1.5x 20-period average.
# Exit when momentum deteriorates (Bull Power <= 0 for longs, Bear Power <= 0 for shorts) or trend fails.
# Uses 6h timeframe with 13-period EMA for Elder Ray and 200-period EMA for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "6h_ElderRay_EMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # EMA200 for trend filter
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema200[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0 (bullish momentum), price > EMA200 (uptrend), volume confirmation
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema200[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0 (bearish momentum), price < EMA200 (downtrend), volume confirmation
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and (close[i] < ema200[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (momentum deterioration) or price < EMA200 (trend failure)
            if (bull_power[i] <= 0) or (close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (momentum deterioration) or price > EMA200 (trend failure)
            if (bear_power[i] <= 0) or (close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals