#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Long: Bull Power > 0 AND Bear Power rising (improving) AND price > 12h EMA(34) AND volume > 1.5x 20-period avg
- Short: Bear Power < 0 AND Bull Power falling (deteriorating) AND price < 12h EMA(34) AND volume > 1.5x 20-period avg
- Exit: Opposite Elder Ray signal OR volume drops below average
- Uses Elder Ray to measure bull/bear power relative to EMA, 12h EMA for trend filter, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (strong bull power with uptrend) and bear markets (strong bear power with downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray calculation
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Rate of change for Elder Ray to detect improvement/deterioration
    bull_power_change = np.diff(bull_power, prepend=bull_power[0])
    bear_power_change = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h HTF data for EMA(34) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 34)  # Need 13 for EMA13, 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND improving AND price > 12h EMA34 AND volume confirmation
            if (bull_power[i] > 0 and 
                bull_power_change[i] > 0 and  # Bull power improving
                close[i] > ema_34_12h_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND deteriorating AND price < 12h EMA34 AND volume confirmation
            elif (bear_power[i] < 0 and 
                  bear_power_change[i] < 0 and  # Bear power deteriorating (more negative)
                  close[i] < ema_34_12h_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR volume drops below average
            if (bull_power[i] <= 0 or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR volume drops below average
            if (bear_power[i] >= 0 or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA13_12hEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0