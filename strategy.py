# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Strategy: Camarilla pivot levels (R1, S1) breakout with 1d trend filter and volume confirmation
# Timeframe: 4h for entry, 1d for trend and pivot calculation
# Why it works: Camarilla levels identify key support/resistance; breakouts with volume and trend
# align with institutional flow. Works in bull (breakouts continue) and bear (breakdowns continue).
# Uses discrete position sizing to minimize fee churn. Target: 20-50 trades/year.

#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # Using previous day's values to avoid look-ahead
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    range_1d = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    r1_1d = np.roll(close_1d, 1) + range_1d * 1.1 / 12
    s1_1d = np.roll(close_1d, 1) - range_1d * 1.1 / 12
    
    # Align to 4h timeframe (will be available after 1d bar closes)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 20-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema34_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R1 with uptrend and volume spike
        long_breakout = (close[i] > r1_4h[i] and close[i] > ema34_4h[i] and volume_spike[i])
        
        # Short conditions: price breaks below S1 with downtrend and volume spike
        short_breakout = (close[i] < s1_4h[i] and close[i] < ema34_4h[i] and volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite pivot level
        elif position == 1 and close[i] < s1_4h[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_4h[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0