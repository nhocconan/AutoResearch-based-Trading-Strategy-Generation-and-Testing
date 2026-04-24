#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 of median price.
  Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling.
- Trend filter: Only trade in direction of 12h EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strong Bull Power in uptrend, in bear via selling strong Bear Power in downtrend.
- Uses EMA for smoothing which reduces whipsaw vs SMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Bull Power = High - EMA13
    bear_power = low - ema_13_6h   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # EMA34 + volume MA + EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h EMA34 trend
            if i > 0 and not np.isnan(ema_34_12h_aligned[i-1]):
                ema34_slope = ema_34_12h_aligned[i] - ema_34_12h_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long when Bull Power > 0 and rising (strong buying pressure)
                    if bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1]) and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short when Bear Power < 0 and falling (strong selling pressure)
                    if bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1]) and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power becomes negative or trend changes
            if bull_power[i] <= 0 or (i > 0 and ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power becomes positive or trend changes
            if bear_power[i] >= 0 or (i > 0 and ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0