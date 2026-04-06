#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 12h EMA50 > EMA200 (bullish trend)
# Short when Bear Power < 0 AND Bull Power > 0 AND 12h EMA50 < EMA200 (bearish trend)
# Exit when Bull Power and Bear Power converge (both near zero)
# Uses 6h timeframe for balance of signal frequency and noise reduction
# Elder Ray captures bull/bear power via EMA13, trend filter avoids counter-trend trades
# Target: 75-150 total trades over 4 years (19-38/year) for optimal 6h performance

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components: EMA13, Bull Power, Bear Power
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA50 and EMA200 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_12h = close_12h_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if required data not available
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Elder Ray convergence (loss of momentum)
        if position == 1:  # long position
            if bull_power[i] < 0 and bear_power[i] > 0:  # powers crossed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] > 0 and bear_power[i] < 0:  # powers crossed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter
            # Long: Bull Power > 0 AND Bear Power < 0 AND 12h EMA50 > EMA200
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_12h_aligned[i] > ema200_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND 12h EMA50 < EMA200
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  ema50_12h_aligned[i] < ema200_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>