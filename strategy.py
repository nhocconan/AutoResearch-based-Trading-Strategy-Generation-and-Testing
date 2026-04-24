#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Williams %R calculation.
- Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
- Entry: Long when Williams %R < -80 (oversold) and rising (current > previous) with volume spike and price > 1d EMA50 (uptrend bias).
         Short when Williams %R > -20 (overbought) and falling (current < previous) with volume spike and price < 1d EMA50 (downtrend bias).
- Exit: When Williams %R crosses above -50 for longs or below -50 for shorts.
- Works in bull via buying oversold dips in uptrend, in bear via selling overbought rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1d data for Williams %R and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
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
            # Check for Williams %R signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Williams %R < -80 (oversold) and rising (current > previous) with uptrend bias
                if williams_r_aligned[i] < -80 and williams_r_aligned[i] > williams_r_aligned[i-1] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) and falling (current < previous) with downtrend bias
                elif williams_r_aligned[i] > -20 and williams_r_aligned[i] < williams_r_aligned[i-1] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (exiting oversold)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (exiting overbought)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0