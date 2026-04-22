#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1-day EMA50 trend filter and volume spike confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Strong Bull Power + price above EMA50 + volume spike = long
# Strong Bear Power + price below EMA50 + volume spike = short
# Designed to capture institutional buying/selling pressure with trend alignment.
# Target: 15-25 trades/year per symbol (60-100 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (using close prices)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Load 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1-day timeframe
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align indicators to 6-hour timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong Bull Power + price above EMA50 + volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Strong Bear Power + price below EMA50 + volume spike
            elif (bear_power_aligned[i] > 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power weakens or price crosses EMA50
            if position == 1:
                if (bull_power_aligned[i] <= 0 or close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (bear_power_aligned[i] <= 0 or close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA50_Trend_Volume_Spike"
timeframe = "6h"
leverage = 1.0