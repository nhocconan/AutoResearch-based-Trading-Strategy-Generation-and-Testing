#!/usr/bin/env python3
"""
4h_Chaikin_Money_Flow_Trend_Confirmation
Hypothesis: Use daily Chaikin Money Flow (CMF) with 4h price action to capture institutional flow.
Long when CMF > +0.15 and price above 20-period EMA.
Short when CMF < -0.15 and price below 20-period EMA.
Exit when CMF crosses zero or price crosses EMA in opposite direction.
Designed for low trade frequency (<50/year) to capture sustained trends with volume confirmation.
Works in both bull and bear markets by following institutional money flow.
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
    
    # === Daily Chaikin Money Flow ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high_1d - low_1d
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / hl_range
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume_1d
    
    # 20-period CMF = Sum(Money Flow Volume, 20) / Sum(Volume, 20)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Align CMF to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # === 4h EMA for trend filter ===
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period: enough for daily CMF and EMA calculations
    warmup = 40  # Covers 20-day CMF + buffer
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: CMF bullish (> +0.15) and price above EMA
            if cmf_aligned[i] > 0.15 and close[i] > ema_20[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: CMF bearish (< -0.15) and price below EMA
            elif cmf_aligned[i] < -0.15 and close[i] < ema_20[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when CMF turns bearish (< 0) or price crosses below EMA
            if cmf_aligned[i] < 0 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when CMF turns bullish (> 0) or price crosses above EMA
            if cmf_aligned[i] > 0 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chaikin_Money_Flow_Trend_Confirmation"
timeframe = "4h"
leverage = 1.0