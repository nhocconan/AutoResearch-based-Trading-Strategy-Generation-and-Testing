#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h EMA50 + Volume Spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Trades in direction of 12h EMA50 trend with volume confirmation.
# Works in bull (buy Bull Power > 0) and bear (sell Bear Power > 0) markets.
# Target: 50-150 total trades over 4 years.
# Timeframe: 6h, HTF: 12h

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray (on 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA50 for trend filter (on 12h)
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Elder Ray components
    bull_power = high - ema13  # Higher = stronger bullish pressure
    bear_power = ema13 - low   # Higher = stronger bearish pressure
    
    # Volume spike detector (20-period median)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long: Bull Power > 0 (bullish pressure) + price above 12h EMA50 + volume spike
        if (bull_power[i] > 0 and
            close[i] > ema50_12h_aligned[i] and
            volume[i] > 2.0 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Bear Power > 0 (bearish pressure) + price below 12h EMA50 + volume spike
        elif (bear_power[i] > 0 and
              close[i] < ema50_12h_aligned[i] and
              volume[i] > 2.0 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Elder Ray power exceeds threshold OR volume dries up
        elif position == 1 and (bear_power[i] > 0.5 * np.std(bull_power[max(0, i-50):i+1]) or volume[i] < 0.5 * vol_median[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] > 0.5 * np.std(bear_power[max(0, i-50):i+1]) or volume[i] < 0.5 * vol_median[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0