#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA trend filter + volume confirmation
# Uses 6h Bull Power (close - EMA13) and Bear Power (EMA13 - close) for momentum
# 12h EMA50 determines trend direction: only take long signals when price > EMA50, short when price < EMA50
# Volume confirmation requires current volume > 1.5x 20-period average to filter weak breakouts
# Works in bull/bear: EMA filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13_6h  # Close - EMA13
    bear_power = ema_13_6h - close  # EMA13 - Close
    
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
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power > 0 (momentum shift) OR price < 12h EMA50 (trend change)
            if bear_power[i] > 0 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 (momentum shift) OR price > 12h EMA50 (trend change)
            if bull_power[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray momentum
            if volume_confirmed:
                # Long entry: Bull Power > 0 AND price > 12h EMA50 (bullish momentum + uptrend)
                if bull_power[i] > 0 and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power > 0 AND price < 12h EMA50 (bearish momentum + downtrend)
                elif bear_power[i] > 0 and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals