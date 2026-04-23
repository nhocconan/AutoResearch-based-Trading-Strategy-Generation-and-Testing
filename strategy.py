#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h, HTF: 4h for trend filter and Camarilla calculation
- Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12 from previous 4h bar
- Long: price breaks above R1 + price > 4h EMA34 (uptrend) + volume > 1.5x 24-period avg
- Short: price breaks below S1 + price < 4h EMA34 (downtrend) + volume > 1.5x 24-period avg
- Exit: price crosses 4h EMA34 (trend-based exit)
- Session filter: 08-20 UTC only (reduces noise trades)
- Uses volume spike (1.5x) to reduce false breakouts, proven effective in ETH/SOL
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (failed breaks reverse)
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
    
    # Volume confirmation: > 1.5x 24-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 4h data ONCE before loop for EMA34 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 4h bar (R1, S1)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Session filter: 08-20 UTC only
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34)  # Need 24 for volume MA, 34 for 4h EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if in_session:
                # Long: price breaks above R1 + price > 4h EMA34 (uptrend) + volume spike
                if volume_spike and close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below S1 + price < 4h EMA34 (downtrend) + volume spike
                elif volume_spike and close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price crosses below 4h EMA34 (trend-based exit)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 4h EMA34 (trend-based exit)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0