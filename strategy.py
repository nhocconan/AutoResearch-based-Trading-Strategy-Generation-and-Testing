#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation.
- Camarilla levels: R4 = close + 1.1*(high-low)*1.1/6, S4 = close - 1.1*(high-low)*1.1/6
- Long: price breaks above R4 + price > 12h EMA50 (uptrend) + volume > 2.0x 20-period avg
- Short: price breaks below S4 + price < 12h EMA50 (downtrend) + volume > 2.0x 20-period avg
- Exit: price retouches 12h EMA50 (trend-based exit) OR opposite signal
- 12h EMA50 provides medium-term trend alignment to reduce counter-trend trades
- Volume confirmation ensures institutional participation in breakouts
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in bull (trend continuation) and bear (mean reversion via faded momentum)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for EMA50 trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (R4, S4)
    # R4 = close + 1.1*(high-low)*1.1/6, S4 = close - 1.1*(high-low)*1.1/6
    camarilla_r4 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 6
    camarilla_s4 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 6
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R4 + price > 12h EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > camarilla_r4_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + price < 12h EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < camarilla_s4_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retouches 12h EMA50 (trend-based exit)
            if close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retouches 12h EMA50 (trend-based exit)
            if close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0