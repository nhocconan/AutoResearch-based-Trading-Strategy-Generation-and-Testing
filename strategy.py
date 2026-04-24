#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend.
- Williams %R(14): Oversold < -80 (long), Overbought > -20 (short).
- Entry: Williams %R extreme + volume > 2.0x 20-period volume MA + price in direction of 1d EMA34.
- Exit: Williams %R returns to neutral range (-50) or opposite extreme.
- Trend filter: Only trade in direction of 1d EMA34 (long if close > EMA34, short if close < EMA34).
- Works in bull via buying oversold pullbacks in uptrend, in bear via selling overbought rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams %R(14) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 + volume MA + Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Williams %R extreme with volume spike and trend filter
            if volume_spike[i]:
                # Long entry: Williams %R < -80 (oversold) and close > 1d EMA34 (uptrend)
                if williams_r[i] < -80 and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: Williams %R > -20 (overbought) and close < 1d EMA34 (downtrend)
                elif williams_r[i] > -20 and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or opposite extreme
            if williams_r[i] > -50:  # Exit when momentum fades
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or opposite extreme
            if williams_r[i] < -50:  # Exit when momentum fades
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0