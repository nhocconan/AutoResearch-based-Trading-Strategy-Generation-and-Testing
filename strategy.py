#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and volume confirmation.
- Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12 (inner levels for 1h precision)
- Long: price breaks above R1 + price > 4h EMA200 (uptrend) + volume > 2.0x 20-period avg
- Short: price breaks below S1 + price < 4h EMA200 (downtrend) + volume > 2.0x 20-period avg
- Exit: price crosses 4h EMA200 (trend-based exit)
- Uses 4h for signal direction (reduces whipsaws), 1h only for entry timing
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Discrete position sizing: ±0.20 to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
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
    
    # Volume confirmation: > 2.0x 20-period average (tight spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 4h data ONCE before loop for EMA200 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA200 for trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate Camarilla levels from previous 4h bar (R1, S1)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 200)  # Need 20 for volume MA, 200 for 4h EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            if in_session and volume_spike:
                # Long: price breaks above R1 + price > 4h EMA200 (uptrend)
                if close[i] > camarilla_r1_aligned[i] and close[i] > ema_200_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below S1 + price < 4h EMA200 (downtrend)
                elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_200_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price crosses below 4h EMA200 (trend-based exit)
            if close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 4h EMA200 (trend-based exit)
            if close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA200_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0