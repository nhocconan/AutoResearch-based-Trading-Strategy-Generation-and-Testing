#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + price > 12h EMA50 (uptrend) + volume > 1.8x 20-period average
- Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) + price < 12h EMA50 (downtrend) + volume > 1.8x 20-period average
- Exit: Momentum divergence (Bull Power < 0 for long, Bear Power < 0 for short) OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag
- Elder Ray measures bull/bear power relative to EMA13; works in both bull and bear markets by identifying momentum extremes
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA50 needs 50, volume MA needs 20, EMA13 needs 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Elder Ray signals with trend filter and volume confirmation
        # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + uptrend + volume spike
        # Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) + downtrend + volume spike
        long_signal = (bull_power[i] > 0 and 
                      bear_power[i] < 0 and
                      uptrend and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (bear_power[i] > 0 and 
                       bull_power[i] < 0 and
                       downtrend and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Momentum divergence OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power turns negative (loss of bullish momentum) or trend turns down
                if (bull_power[i] <= 0 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power turns negative (loss of bearish momentum) or trend turns up
                if (bear_power[i] <= 0 or 
                    not downtrend):
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