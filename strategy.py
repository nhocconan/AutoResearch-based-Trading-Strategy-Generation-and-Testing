#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 12h EMA200 trend filter + volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 12h EMA200 provides higher timeframe trend direction to avoid counter-trend trades
# Volume confirmation ensures breakout authenticity
# Discrete sizing 0.25 limits drawdown during 2022 crash while capturing trends
# Works in bull/bear: trend filter adapts, Elder Ray captures momentum shifts
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing

name = "6h_12h_elder_ray_ema200_volume_v1"
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
    
    # Load 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA200
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirmed = volume[i] > 1.5 * avg_volume
        else:
            volume_confirmed = False
        
        if position == 1:  # Long position
            # Exit: bear power turns positive (selling pressure weakening) OR trend turns bearish
            if bear_power[i] > 0 or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power turns negative (buying pressure weakening) OR trend turns bullish
            if bull_power[i] < 0 or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray + 12h EMA filter
            if volume_confirmed:
                # Long entry: strong bull power AND price above 12h EMA200 (bullish alignment)
                if bull_power[i] > 0 and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: strong bear power AND price below 12h EMA200 (bearish alignment)
                elif bear_power[i] < 0 and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals