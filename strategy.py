#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Swing_Double_EMA"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 and EMA21
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    bull_power = high_1d - ema13_1d  # Bull Power: High - EMA13
    bear_power = low_1d - ema21_1d   # Bear Power: Low - EMA21
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13[i]) or np.isnan(ema21[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA13 > EMA21 (bullish trend) + Bull Power > 0 (strong buying) + volume filter
            long_cond = (ema13[i] > ema21[i]) and \
                        (bull_power_aligned[i] > 0) and \
                        volume_filter[i]
            # Short: EMA13 < EMA21 (bearish trend) + Bear Power < 0 (strong selling) + volume filter
            short_cond = (ema13[i] < ema21[i]) and \
                         (bear_power_aligned[i] < 0) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA13 < EMA21 (trend reversal) or Bull Power <= 0 (buying pressure gone)
            if (ema13[i] < ema21[i]) or (bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA13 > EMA21 (trend reversal) or Bear Power >= 0 (selling pressure gone)
            if (ema13[i] > ema21[i]) or (bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals