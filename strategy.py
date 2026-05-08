#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend and volume confirmation
# Uses daily EMA50 trend filter and volume spikes to capture high-probability reversals
# Works in both bull and bear markets by fading extremes with trend alignment
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Calculate Elder Ray components (Bull Power = High - EMA, Bear Power = Low - EMA)
    # Use 13-period EMA for Elder Ray calculation (standard)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bear Power negative (bearish pressure) but price above 1d EMA50 with volume spike
            # This indicates potential reversal from bearish pressure in uptrend
            if (bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power positive (bullish pressure) but price below 1d EMA50 with volume spike
            # This indicates potential reversal from bullish pressure in downtrend
            elif (bull_power[i] > 0 and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns positive (bullish pressure returns) or trend fails
            if (bull_power[i] > 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns negative (bearish pressure returns) or trend fails
            if (bear_power[i] < 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals