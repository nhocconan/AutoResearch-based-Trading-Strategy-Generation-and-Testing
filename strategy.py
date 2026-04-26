#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Enters long when price breaks above 20-day high, close > 1w EMA50, and volume spike (>2x 20-day avg).
Enters short when price breaks below 20-day low, close < 1w EMA50, and volume spike.
Exits on opposite breakout or trend reversal (close crosses 1w EMA50).
Uses 1d primary timeframe to target 7-25 trades/year (30-100 total over 4 years).
Donchian channels provide clear structure, 1w EMA filters counter-trend trades in bear markets,
volume confirmation reduces false breakouts. Works in bull/bear markets by aligning with weekly trend.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align to 1d timeframe with proper delay (wait for 1w bar close)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-day Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 50 for 1w EMA alignment)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price > 20-day high, close > 1w EMA50, volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < 20-day low, close < 1w EMA50, volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < 20-day low OR close < 1w EMA50 (trend reversal)
            if (close[i] < low_roll[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > 20-day high OR close > 1w EMA50 (trend reversal)
            if (close[i] > high_roll[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0