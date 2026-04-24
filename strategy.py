#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Williams %R calculation.
- Williams %R(14) measures momentum extremes: < -80 = oversold, > -20 = overbought.
- Entry: Long when Williams %R crosses above -80 from below with volume spike and close > 1d EMA50 (uptrend).
         Short when Williams %R crosses below -20 from above with volume spike and close < 1d EMA50 (downtrend).
- Exit: When Williams %R returns to opposite extreme (-20 for long, -80 for short) or volume spike ends.
- Works in bull via buying oversold bounces in uptrend, in bear via selling overbought bounces in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:  # Avoid division by zero
            williams_r[i] = (hh - df_1d['close'].iloc[i]) / (hh - ll) * -100
        else:
            williams_r[i] = -50.0  # Neutral when range is zero
    
    # Align 1d indicators to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for extreme reversal signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish reversal: Williams %R crosses above -80 from below
                if (i > 0 and williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif (i > 0 and williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -20 (overbought) or trend fails
            if williams_r_aligned[i] >= -20 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -80 (oversold) or trend fails
            if williams_r_aligned[i] <= -80 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0