#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Breakouts above/below these levels with volume confirmation capture institutional flow.
# Works in bull markets (long breakouts) and bear markets (short breakdowns).
# Uses volume filter to avoid false breakouts and maintain low trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    # Resistance levels
    R4 = prev_close + (prev_high - prev_low) * 1.5000
    R3 = prev_close + (prev_high - prev_low) * 1.2500
    R2 = prev_close + (prev_high - prev_low) * 1.1666
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    # Support levels
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    S2 = prev_close - (prev_high - prev_low) * 1.1666
    S3 = prev_close - (prev_high - prev_low) * 1.2500
    S4 = prev_close - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Breakout signals
        breakout_above_R4 = close[i] > R4_aligned[i-1]
        breakdown_below_S4 = close[i] < S4_aligned[i-1]
        
        # Entry conditions
        # Long: Breakout above R4 with volume confirmation
        if breakout_above_R4 and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below S4 with volume confirmation
        elif breakdown_below_S4 and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Reversion to mean - close opposite Camarilla level
        elif position == 1 and close[i] < S1_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > R1_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals