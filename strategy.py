# 1/1
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Elder Ray Index with regime filter.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 (bullish divergence) AND price > EMA50 (uptrend).
# Short when Bear Power > 0 and Bull Power < 0 (bearish divergence) AND price < EMA50 (downtrend).
# Uses EMA50 as trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by trading with the EMA50 trend.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Elder Ray and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA13 and EMA50
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_1d - ema13  # High - EMA13
    bear_power = ema13 - low_1d   # EMA13 - Low
    
    # Calculate EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # Need EMA50 period
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for Elder Ray signals with EMA50 trend filter
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish divergence) AND price > EMA50
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish divergence) AND price < EMA50
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Elder Ray turns bearish OR price crosses below EMA50
            if (bull_power_aligned[i] <= 0 or 
                bear_power_aligned[i] >= 0 or
                close[i] <= ema50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Elder Ray turns bullish OR price crosses above EMA50
            if (bear_power_aligned[i] <= 0 or 
                bull_power_aligned[i] >= 0 or
                close[i] >= ema50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_EMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0