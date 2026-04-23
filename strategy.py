#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 Trend Filter and Volume Confirmation
- Long: Bull Power > 0 (close > EMA13) + Bear Power < 0 (close < EMA13) + price > 12h EMA50 (uptrend) + volume > 1.5x 20-period average
- Short: Bear Power < 0 (close < EMA13) + Bull Power > 0 (close > EMA13) + price < 12h EMA50 (downtrend) + volume > 1.5x 20-period average
- Exit: Opposite Elder Ray signal (Bull Power < 0 for longs exit, Bear Power > 0 for shorts exit)
- Uses Elder Ray to measure bull/bear strength relative to EMA13, 12h EMA50 for trend filter
- Volume confirmation ensures institutional participation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
- Works in both bull and bear markets by capturing momentum shifts
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
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema13  # > 0 when close > EMA13 (bullish)
    bear_power = close - ema13  # < 0 when close < EMA13 (bearish) - same calculation, interpretation different
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20)  # EMA13 needs 13, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Elder Ray signals with trend filter and volume confirmation
        # Long: Bull Power > 0 (bullish momentum) + Bear Power < 0 (not bearish) + uptrend + volume spike
        # Actually: Bull Power > 0 means bullish, Bear Power < 0 means bearish
        # We want: Bull Power > 0 AND Bear Power < 0 is impossible since they're the same
        # Correct interpretation: Bull Power > 0 = bullish pressure, Bear Power < 0 = bearish pressure
        # Long: Bull Power > 0 (bulls in control) + uptrend + volume spike
        # Short: Bear Power < 0 (bears in control) + downtrend + volume spike
        long_signal = (bull_power[i] > 0 and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (bear_power[i] < 0 and 
                       downtrend and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Opposite Elder Ray signal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bulls lose control (Bull Power <= 0)
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bears lose control (Bear Power >= 0)
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0