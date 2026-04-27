#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 1d trend filter and volume confirmation.
# Uses daily Bull Power (high - EMA13) and Bear Power (EMA13 - low) to detect institutional buying/selling pressure.
# Long when Bull Power > 0 and Bear Power < 0 (bullish bias) with 1d uptrend and volume spike.
# Short when Bear Power > 0 and Bull Power < 0 (bearish bias) with 1d downtrend and volume spike.
# Designed for 12-30 trades/year per symbol (48-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = high - EMA13, Bear Power = EMA13 - low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe (wait for daily bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0 AND Bear Power < 0 (bullish bias) AND 1d uptrend AND volume spike
        if (bull_power_aligned[i] > 0 and 
            bear_power_aligned[i] < 0 and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Bear Power > 0 AND Bull Power < 0 (bearish bias) AND 1d downtrend AND volume spike
        elif (bear_power_aligned[i] > 0 and 
              bull_power_aligned[i] < 0 and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0