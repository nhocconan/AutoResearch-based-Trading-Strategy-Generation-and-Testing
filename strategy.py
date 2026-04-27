#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_DailyTrend_VolumeSpike
Hypothesis: Price touching Camarilla R3 or S3 levels on 12h with daily trend alignment and volume spike captures mean reversion in ranging markets and breakouts in trending markets. Works in bull/bear via volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous day's values
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d_prev, 1)
    high_prev[0] = high_1d[0]  # first bar uses same day
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d_prev[0]
    
    rng = high_prev - low_prev
    r3 = close_prev + (rng * 1.1 / 4)
    s3 = close_prev - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppy market filter: use 12h ATR ratio
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    chop_filter = atr / (atr_ma + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid extreme volatility (chop > 2.0) or too quiet (chop < 0.5)
        if chop_filter[i] > 2.0 or chop_filter[i] < 0.5:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or crosses S3 with volume spike and daily uptrend
            if (close[i] <= s3_aligned[i] and volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses R3 with volume spike and daily downtrend
            elif (close[i] >= r3_aligned[i] and volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses above midpoint (C) or trend fails
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= midpoint or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below midpoint (C) or trend fails
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= midpoint or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0