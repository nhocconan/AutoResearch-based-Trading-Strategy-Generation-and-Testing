#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d trend filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and 1d EMA50 up; Short when Bear Power > 0 and 1d EMA50 down
# Works in bull (follows trend) and bear (countertrend reversals at extremes)
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
name = "6h_ElderRay_BullPower_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Elder Ray and daily EMA50 to 6h
    bull_power_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bear_power)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive and 1d trend up (EMA50 rising)
            if bull_val > 0 and (i == start_idx or ema50_val > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive and 1d trend down (EMA50 falling)
            elif bear_val > 0 and (i == start_idx or ema50_val < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or 1d trend turns down
            if bull_val <= 0 or (i > start_idx and ema50_val < ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns negative or 1d trend turns up
            if bear_val <= 0 or (i > start_idx and ema50_val > ema50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals