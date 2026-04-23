#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Entry: Price breaks above R1 (bullish) or below S1 (bearish) with volume > 2x 20-period average
- Trend filter: price > 1d EMA34 for longs, price < 1d EMA34 for shorts
- Exit: Opposite Camarilla breakout (price crosses below S1 for longs, above R1 for shorts)
- Uses Camarilla pivot levels for intraday support/resistance, EMA34 for HTF trend alignment
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by EMA34)
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
    
    # Volume confirmation: > 2.0x 20-period average to reduce trades
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Need previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_range
    s1 = prev_close - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need 34 for EMA34, 20 for volume MA, +1 for previous day data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average to reduce trade frequency)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R1 + price > 1d EMA34 + volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + price < 1d EMA34 + volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 (opposite Camarilla level)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1 (opposite Camarilla level)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0