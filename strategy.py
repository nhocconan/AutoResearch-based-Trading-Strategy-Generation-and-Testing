#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d EMA200 regime filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; 1d EMA200 defines long-term trend
# In bull regime (price > 1d EMA200): long when bull power > 0 and rising
# In bear regime (price < 1d EMA200): short when bear power < 0 and falling
# Volume confirms conviction; discrete sizing 0.25 limits drawdown
# Works in bull/bear: regime filter adapts, Elder Ray captures strength/weakness
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing

name = "6h_12h_elder_ray_ema_volume_v1"
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
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])  # SMA seed
        alpha = 2.0 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200[i] = alpha * close_1d[i] + (1 - alpha) * ema_200[i-1]
    
    # Align 1d EMA200 to 6h timeframe (wait for 1d bar close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = np.full(n, np.nan)
    if len(close) >= 13:
        ema_13[12] = np.mean(close[:13])  # SMA seed
        alpha = 2.0 / (13 + 1)
        for i in range(13, n):
            ema_13[i] = alpha * close[i] + (1 - alpha) * ema_13[i-1]
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA13
    bear_power = low - ema_13   # Bear power: low minus EMA13
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
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
            # Exit: bear power >= 0 (bulls losing control) OR price < 1d EMA200 (regime change)
            if bear_power[i] >= 0 or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power <= 0 (bears losing control) OR price > 1d EMA200 (regime change)
            if bull_power[i] <= 0 or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray + 1d EMA200 filter
            if volume_confirmed:
                # Long entry: price > 1d EMA200 (bull regime) AND bull power > 0 AND rising
                if (close[i] > ema_200_aligned[i] and 
                    bull_power[i] > 0 and 
                    i >= 101 and bull_power[i] > bull_power[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < 1d EMA200 (bear regime) AND bear power < 0 AND falling
                elif (close[i] < ema_200_aligned[i] and 
                      bear_power[i] < 0 and 
                      i >= 101 and bear_power[i] < bear_power[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals