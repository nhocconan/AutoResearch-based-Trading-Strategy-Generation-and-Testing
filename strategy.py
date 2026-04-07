#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Camarilla Pivot with Volume Confirmation
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from weekly data act as strong
# support/resistance. Price breaking above R4 or below S4 with volume confirms
# institutional breakout, while rejection at R3/S3 with volume indicates fade.
# Works in both bull and bear markets: breakouts capture trends, fades capture
# reversals in ranging conditions. Weekly timeframe filters noise.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "6h_weekly_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate Camarilla levels for current week (use previous week's data)
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    weekly_range = weekly_high - weekly_low
    r4 = weekly_close + (weekly_range * 1.1 / 2)
    r3 = weekly_close + (weekly_range * 1.1 / 4)
    s3 = weekly_close - (weekly_range * 1.1 / 4)
    s4 = weekly_close - (weekly_range * 1.1 / 2)
    
    # Shift by 1 to use previous week's levels (avoid look-ahead)
    prev_r4 = np.roll(r4, 1)
    prev_r3 = np.roll(r3, 1)
    prev_s3 = np.roll(s3, 1)
    prev_s4 = np.roll(s4, 1)
    # Handle first element
    if len(prev_r4) > 1:
        prev_r4[0] = prev_r4[1]
        prev_r3[0] = prev_r3[1]
        prev_s3[0] = prev_s3[1]
        prev_s4[0] = prev_s4[1]
    else:
        prev_r4[0] = 0
        prev_r3[0] = 0
        prev_s3[0] = 0
        prev_s4[0] = 0
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, prev_r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, prev_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, prev_s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, prev_s4)
    
    # Volume filter: volume > 1.8x 24-period average (4 days of 6h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S3 or breaks below S4 (failed breakout)
            if low[i] < s3_aligned[i] or low[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R3 or breaks above R4 (failed breakdown)
            if high[i] > r3_aligned[i] or high[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long breakout: price breaks above R4 with volume
            if (high[i] > r4_aligned[i] or close[i] > r4_aligned[i]) and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S4 with volume
            elif (low[i] < s4_aligned[i] or close[i] < s4_aligned[i]) and vol_filter[i]:
                position = -1
                signals[i] = -0.25
            # Long fade: price rejects from R3 with volume (mean reversion)
            elif (high[i] > r3_aligned[i] and close[i] < r3_aligned[i]) and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short fade: price rejects from S3 with volume (mean reversion)
            elif (low[i] < s3_aligned[i] and close[i] > s3_aligned[i]) and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals