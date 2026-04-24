#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Williams %R calculation.
- Williams %R calculated on 1d timeframe: values below -80 = oversold, above -20 = overbought.
- Entry: Long when Williams %R crosses above -80 from below with volume spike and close > 1d EMA50.
         Short when Williams %R crosses below -20 from above with volume spike and close < 1d EMA50.
- Exit: When Williams %R returns to opposite extreme (-20 for longs, -80 for shorts) or volume dries up.
- Works in bull via buying oversold bounces in uptrend, in bear via selling overbought retraces in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
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
    williams_r = williams_r.fillna(-50)  # Neutral value when range is zero
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
    
    # Get 1d data for Williams %R and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for Williams %R(14) and EMA(50)
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Williams %R(14)
    williams_r = calculate_williams_r(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
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
            # Check for entry signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish entry: Williams %R crosses above -80 from below and close > EMA50
                if (williams_r_aligned[i] > -80 and 
                    i > start_idx and williams_r_aligned[i-1] <= -80 and
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above and close < EMA50
                elif (williams_r_aligned[i] < -20 and 
                      i > start_idx and williams_r_aligned[i-1] >= -20 and
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -20 (overbought) or loss of volume spike
            if williams_r_aligned[i] >= -20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -80 (oversold) or loss of volume spike
            if williams_r_aligned[i] <= -80 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0