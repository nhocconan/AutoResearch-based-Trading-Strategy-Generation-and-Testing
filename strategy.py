#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation.
- Camarilla R1/S1 levels from previous 4h bar: S1 = close - 1.12*(high-low)/12, R1 = close + 1.12*(high-low)/12
- Long: price breaks above R1 + volume > 1.5x 20-period avg + price > 4h EMA50
- Short: price breaks below S1 + volume > 1.5x 20-period avg + price < 4h EMA50
- Exit: Opposite Camarilla level touch (S1 for long, R1 for short) or 4h EMA50 trend flip
- Uses Camarilla for intraday support/resistance, volume for conviction, 4h EMA50 for HTF trend filter
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get previous completed 4h bar for Camarilla calculation
        # We need the 4h bar that closed before the current 1h bar
        # Since we're on 1h timeframe, we look back to find the last completed 4h bar
        htf_idx = i // 4  # Each 4h bar contains 4 1h bars
        if htf_idx < 1:  # Need at least one previous 4h bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Use previous completed 4h bar (htf_idx-1) to avoid look-ahead
        prev_htf_idx = htf_idx - 1
        if prev_htf_idx >= 0 and prev_htf_idx < len(df_4h):
            phigh = high_4h[prev_htf_idx]
            plow = low_4h[prev_htf_idx]
            pclose = close_4h[prev_htf_idx]
            
            # Calculate Camarilla levels from previous 4h bar
            range_ = phigh - plow
            if range_ <= 0:
                # Avoid division by zero, use previous bar's close as fallback
                r1 = pclose
                s1 = pclose
            else:
                r1 = pclose + 1.12 * range_ / 12
                s1 = pclose - 1.12 * range_ / 12
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + price > 4h EMA50
            if (close[i] > r1 and 
                volume_confirm and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume confirmation + price < 4h EMA50
            elif (close[i] < s1 and 
                  volume_confirm and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price touches S1 OR price < 4h EMA50 (trend flip)
            if close[i] <= s1 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price touches R1 OR price > 4h EMA50 (trend flip)
            if close[i] >= r1 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0