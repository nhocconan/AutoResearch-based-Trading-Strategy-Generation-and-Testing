#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation. 
Long when price breaks above R1 + above 1d EMA50 + volume > 1.5x 20-period average.
Short when price breaks below S1 + below 1d EMA50 + volume > 1.5x 20-period average.
Uses discrete sizing (0.25) to minimize fees. Designed for 12-37 trades/year on 12h timeframe.
Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h bar (using typical price)
    # Camarilla levels based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    R1 = typical_price + (range_hl * 1.1 / 12)
    S1 = typical_price - (range_hl * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA50 + volume spike
            long_signal = (close[i] > R1[i]) and (close[i] > ema_50_aligned[i]) and volume_spike
            # Short: price breaks below S1 + below 1d EMA50 + volume spike
            short_signal = (close[i] < S1[i]) and (close[i] < ema_50_aligned[i]) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S1 (breakdown below support)
            exit_signal = close[i] < S1[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R1 (breakout above resistance)
            exit_signal = close[i] > R1[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0