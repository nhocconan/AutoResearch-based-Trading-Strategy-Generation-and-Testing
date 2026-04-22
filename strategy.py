#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Exponential Moving Average (EMA) crossover strategy with daily Supertrend filter and volume confirmation
# EMA(9)/EMA(21) crossover on 12h chart captures medium-term momentum shifts
# Daily Supertrend (ATR=10, multiplier=3) ensures trades align with higher-timeframe trend
# Volume > 1.3x 20-period average confirms momentum strength
# Designed to work in both bull and bear markets by following the daily trend direction
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining responsiveness

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Supertrend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    upper_band = (high_1d + low_1d) / 2 + 3 * atr
    lower_band = (high_1d + low_1d) / 2 - 3 * atr
    
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 12h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # EMA crossover on 12h data
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Wait for slow EMA
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish crossover: fast EMA crosses above slow EMA
            bullish_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
            # Bearish crossover: fast EMA crosses below slow EMA
            bearish_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
            
            # Long: Bullish crossover + daily uptrend + volume spike
            if bullish_cross and supertrend_direction_aligned[i] == 1 and volume[i] > 1.3 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish crossover + daily downtrend + volume spike
            elif bearish_cross and supertrend_direction_aligned[i] == -1 and volume[i] > 1.3 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Opposite crossover or trend change
            if position == 1:
                # Exit long: Bearish crossover or daily downtrend
                bearish_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
                if bearish_cross or supertrend_direction_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bullish crossover or daily uptrend
                bullish_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
                if bullish_cross or supertrend_direction_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_EMACrossover_1dSupertrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0