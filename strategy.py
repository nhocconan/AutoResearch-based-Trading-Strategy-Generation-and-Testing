#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R1/S1 breakout on 1d timeframe with 1w EMA50 trend filter and volume confirmation (>1.5x 20-period MA). 
Long when price breaks above R1 with 1w uptrend and volume spike. 
Short when price breaks below S1 with 1w downtrend and volume spike.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the 1w trend, which adapts to regime changes.
Target: 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar: use current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA + 20 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend and volume spike
            if (close[i] > r1[i] and uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1w downtrend and volume spike
            elif (close[i] < s1[i] and downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1w trend changes to downtrend OR price closes below R1 (failed breakout)
            if (not uptrend_1w[i] or close[i] < r1[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1w trend changes to uptrend OR price closes above S1 (failed breakout)
            if (not downtrend_1w[i] or close[i] > s1[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0