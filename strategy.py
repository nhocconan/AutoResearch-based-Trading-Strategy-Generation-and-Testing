#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend.
- Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
- Extreme readings: %R < -80 (oversold) for longs, %R > -20 (overbought) for shorts.
- Entry: Long when %R crosses above -80 from below with volume spike and price > 1w EMA50 (uptrend).
         Short when %R crosses below -20 from above with volume spike and price < 1w EMA50 (downtrend).
- Exit: When %R returns to neutral zone (-50) or opposite signal.
- Works in bull via buying oversold dips in uptrend, in bear via selling overbought rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    highest_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1w['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Williams %R extreme signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Williams %R crosses above -80 from below (oversold recovery) with uptrend
                if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above (overbought rejection) with downtrend
                elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral zone (-50) or breaks above -20 (overbought)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral zone (-50) or breaks below -80 (oversold)
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0