#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume spike confirmation
# Long when price breaks above R3 with volume > 1.8x 20-bar average
# Short when price breaks below S3 with volume > 1.8x 20-bar average
# Exit when price returns to the 1d VWAP (mean reversion to fair value)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R3/S3 represent strong intraday support/resistance levels
# Volume confirmation validates breakout strength while limiting overtrading
# VWAP exit provides timely mean reversion in ranging markets
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "6h_Camarilla_R3_S3_Breakout_1dVWAP_Exit_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot and VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed 1d bars for pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # R3 = Pivot + (Range * 1.1/2)
    # S3 = Pivot - (Range * 1.1/2)
    r3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d VWAP: Typical Price * Volume / Cumulative Volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    # Handle division by zero on first bar
    vwap_1d = np.where(df_1d['volume'].values.cumsum() == 0, typical_price_1d, vwap_1d)
    
    # Align 1d VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike, in session
            if close[i] > r3_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike, in session
            elif close[i] < s3_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP (mean reversion)
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP (mean reversion)
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals