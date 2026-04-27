#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Trend_Volume
Hypothesis: Camarilla R3/S3 breakout on 4h with 12h trend filter and volume spike.
- Camarilla levels calculated from prior 12h OHLC
- Long: price breaks above R3 with volume spike and 12h uptrend (close > EMA34)
- Short: price breaks below S3 with volume spike and 12h downtrend (close < EMA34)
- Exit: price returns to Camarilla Pivot (mean reversion) or trend failure
- Designed to capture medium-term momentum with institutional levels
- Target: 20-50 trades/year (80-200 total over 4 years)
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
    
    # Calculate Camarilla levels from prior 12h OHLC
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Use prior 12h close for Camarilla calculation (avoid look-ahead)
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Camarilla multipliers
    R3 = prev_close + (prev_high - prev_low) * 1.1/2
    S3 = prev_close - (prev_high - prev_low) * 1.1/2
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    Pivot_aligned = align_htf_to_ltf(prices, df_12h, Pivot)
    
    # 12h trend filter (EMA34)
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
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
            np.isnan(Pivot_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 12h uptrend
            if (close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and 12h downtrend
            elif (close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Pivot or trend failure
            if (close[i] <= Pivot_aligned[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Pivot or trend failure
            if (close[i] >= Pivot_aligned[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Trend_Volume"
timeframe = "4h"
leverage = 1.0