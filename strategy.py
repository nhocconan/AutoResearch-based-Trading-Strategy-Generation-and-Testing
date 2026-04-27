#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_DailyTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d trend filter and volume spike.
- Camarilla levels calculated from prior 1d OHLC
- Long: price breaks above R3 with volume spike and 1d uptrend (close > EMA34)
- Short: price breaks below S3 with volume spike and 1d downtrend (close < EMA34)
- Exit: price returns to Camarilla Pivot (mean reversion) or trend failure
- Designed to capture intraday momentum with institutional levels
- Target: 12-30 trades/year (48-120 total over 4 years)
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
    
    # Calculate Camarilla levels from prior 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior 1d close for Camarilla calculation (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla multipliers
    R3 = prev_close + (prev_high - prev_low) * 1.1/2
    S3 = prev_close - (prev_high - prev_low) * 1.1/2
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # 1d trend filter (EMA34)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 1d uptrend
            if (close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and 1d downtrend
            elif (close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Pivot or trend failure
            if (close[i] <= Pivot_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Pivot or trend failure
            if (close[i] >= Pivot_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0