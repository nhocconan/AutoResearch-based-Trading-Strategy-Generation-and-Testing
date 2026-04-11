#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot breakout with 12h volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (based on prior 12h candle) act as strong intraday support/resistance.
# Breakouts above/below these levels with above-average volume capture momentum moves.
# Works in both bull and bear markets as pivots adapt to recent price action.
# Target: 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 12h candle
    # Typical price = (H + L + C) / 3
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use R3, R4, S3, S4 for breakouts
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r4 = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_r3 = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s4 = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 12h volume average (10-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_10_12h)
    
    # Align raw 12h volume for confirmation
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or \
           np.isnan(vol_avg_10_12h_aligned[i]) or np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.2x 10-period average
        vol_confirm = vol_12h_aligned[i] > 1.2 * vol_avg_10_12h_aligned[i]
        
        # Price relative to Camarilla levels
        above_r4 = close[i] > r4_aligned[i]
        above_r3 = close[i] > r3_aligned[i]
        below_s3 = close[i] < s3_aligned[i]
        below_s4 = close[i] < s4_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above R4 with volume confirmation
        if above_r4 and vol_confirm and position != 1:
            # Additional check: ensure we didn't just break above R4 in previous bar
            if i == 10 or close[i-1] <= r4_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price breaks below S4 with volume confirmation
        elif below_s4 and vol_confirm and position != -1:
            # Additional check: ensure we didn't just break below S4 in previous bar
            if i == 10 or close[i-1] >= s4_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to R3/S3 levels (mean reversion)
        elif position == 1 and close[i] < r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals