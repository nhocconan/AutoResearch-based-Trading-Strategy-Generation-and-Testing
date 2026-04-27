# The hypothesis: In both bull and bear markets, weekly (1w) price extremes act as strong support/resistance.
# Price often reverses or pauses when retesting the prior week's high or low, especially with volume confirmation.
# We use the prior week's high and low as dynamic support/resistance levels.
# Entry: Long when price crosses above prior week's high with volume spike; Short when price crosses below prior week's low with volume spike.
# Exit: Mean reversion to the weekly midpoint (average of prior week's high and low).
# Timeframe: 6h to capture swings while avoiding excessive noise and overtrading.
# Weekly levels are updated only once per week, reducing whipsaw and providing institutional-grade levels.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for support/resistance levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's high, low, and midpoint
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    week_mid = (week_high + week_low) / 2.0
    
    # Align weekly levels to 6h timeframe (using prior week's values to avoid look-ahead)
    week_high_aligned = align_htf_to_ltf(prices, df_1w, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1w, week_low)
    week_mid_aligned = align_htf_to_ltf(prices, df_1w, week_mid)
    
    # Volume filter: volume > 1.8x 20-period average (to ensure conviction)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(week_high_aligned[i]) or np.isnan(week_low_aligned[i]) or 
            np.isnan(week_mid_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price crosses above prior week's high with volume spike
        if (close[i] > week_high_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price crosses below prior week's low with volume spike
        elif (close[i] < week_low_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly midpoint (mean reversion)
        elif position == 1 and close[i] < week_mid_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > week_mid_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyHighLow_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0