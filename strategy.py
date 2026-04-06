#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h trend filter
# Long when Bear Power < 0 and 12h EMA(50) rising
# Short when Bull Power > 0 and 12h EMA(50) falling
# Uses EMA(13) for Bull/Bear Power calculation to capture momentum.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components (13-period EMA)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h trend filter: EMA(50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # EMA slope (rising/falling) - 3-period change
    ema_slope = np.zeros_like(ema50_12h_aligned)
    ema_slope[3:] = ema50_12h_aligned[3:] - ema50_12h_aligned[:-3]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if EMA data not available
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(ema_slope[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: trend reversal or power divergence
        if position == 1:  # long position
            # Exit: Bear Power turns positive OR EMA slope turns negative
            if (bear_power[i] >= 0 or ema_slope[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns negative OR EMA slope turns positive
            if (bull_power[i] <= 0 or ema_slope[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment
            # Long: Bear Power negative AND EMA slope rising
            if (bear_power[i] < 0 and ema_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power positive AND EMA slope falling
            elif (bull_power[i] > 0 and ema_slope[i] < 0):
                signals[i] = -0.25
                position = -1
    
    return signals