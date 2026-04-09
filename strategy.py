#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 trend filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA200 defines long-term trend: only take longs above EMA200, shorts below EMA200
# Volume confirmation ensures breakout authenticity
# Works in bull/bear: trend filter adapts, Elder Ray captures momentum shifts
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h timeframe (wait for 1d bar close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative OR price < 1d EMA200 (trend change)
            if bear_power[i] >= 0 or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR price > 1d EMA200 (trend change)
            if bull_power[i] <= 0 or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray + 1d EMA200 filter
            if volume_confirmed:
                # Long entry: Bull Power > 0 AND price > 1d EMA200 (bullish alignment)
                if bull_power[i] > 0 and close[i] > ema_200_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power < 0 AND price < 1d EMA200 (bearish alignment)
                elif bear_power[i] < 0 and close[i] < ema_200_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals