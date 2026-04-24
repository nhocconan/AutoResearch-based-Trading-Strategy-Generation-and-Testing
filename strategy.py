#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1w EMA50 Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend filter.
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- Entry: Long when Williams %R crosses above -80 from below with volume spike and weekly close > weekly EMA50 (uptrend).
         Short when Williams %R crosses below -20 from above with volume spike and weekly close < weekly EMA50 (downtrend).
- Exit: When Williams %R returns to opposite extreme threshold (-20 for longs, -80 for shorts) or mean reversion.
- Works in bull via buying oversold bounces in uptrend, in bear via selling overbought bounces in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = williams_r.fillna(-50)  # neutral value when range is zero
    return williams_r.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 6h data
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Align 1w EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for reversal signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish reversal: Williams %R crosses above -80 from below
                if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above
                elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -20 (overbought threshold) or trend fails
            if williams_r[i] >= -20 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -80 (oversold threshold) or trend fails
            if williams_r[i] <= -80 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0