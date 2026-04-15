#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h EMA50 + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA. Trades with the trend (EMA direction) when power confirms.
# Works in bull (buy on bull power + uptrend) and bear (sell on bear power + downtrend).
# Volume filter ensures conviction. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 (trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe (will be available after 12h bar closes)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 13-period EMA for Elder Ray (on 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA
    bear_power = low - ema_13   # Bear power: low minus EMA
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if EMA50 not yet available (first 50 periods of 12h data)
        if np.isnan(ema_50_aligned[i]):
            continue
        
        # Long: bull power positive (bulls in control) + price above 12h EMA50 (uptrend) + volume confirmation
        if (bull_power[i] > 0 and 
            close[i] > ema_50_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: bear power negative (bears in control) + price below 12h EMA50 (downtrend) + volume confirmation
        elif (bear_power[i] < 0 and 
              close[i] < ema_50_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend change or power divergence
        elif position == 1 and (close[i] < ema_50_aligned[i] or bull_power[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_50_aligned[i] or bear_power[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0