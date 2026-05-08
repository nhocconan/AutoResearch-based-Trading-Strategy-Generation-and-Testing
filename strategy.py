#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power + 1d EMA trend filter with volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > EMA50 (1d trend) AND volume > 1.5x average.
# Short when Bear Power > 0 (low > EMA13) AND Bull Power < 0 (close < EMA13) AND price < EMA50 (1d trend) AND volume > 1.5x average.
# Exit when power signals weaken or trend reverses.
# Uses Elder Ray to measure bull/bear strength relative to EMA, filtered by higher timeframe trend.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_ElderRay_1dEMA_Volume"
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
    
    # EMA13 for Elder Ray calculation (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # EMA50 for trend filter (6h)
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMAs
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, price > EMA50 (6h), price > EMA50 (1d), volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema50[i]) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0, price < EMA50 (6h), price < EMA50 (1d), volume spike
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and (close[i] < ema50[i]) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR price < EMA50 (6h) OR price < EMA50 (1d)
            exit_cond = (bull_power[i] <= 0) or (bear_power[i] >= 0) or (close[i] < ema50[i]) or (close[i] < ema50_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR Bull Power >= 0 OR price > EMA50 (6h) OR price > EMA50 (1d)
            exit_cond = (bear_power[i] <= 0) or (bull_power[i] >= 0) or (close[i] > ema50[i]) or (close[i] > ema50_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals