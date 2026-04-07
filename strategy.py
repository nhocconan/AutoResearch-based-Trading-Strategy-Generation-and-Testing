#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Price breaking above/below weekly pivot support/resistance levels
# on 1d timeframe with volume > 1.5x average indicates institutional interest.
# Weekly pivot levels act as significant support/resistance in both bull and bear markets.
# Volume confirmation filters false breakouts. Target: 10-25 trades/year.

name = "1d_weekly_pivot_range_breakout_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    support1 = 2 * pivot - weekly_high
    resistance1 = 2 * pivot - weekly_low
    
    # Align weekly levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_weekly, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_weekly, resistance1)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(support1_aligned[i]) or np.isnan(resistance1_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below pivot or volume drops
            if close[i] < pivot_aligned[i] or volume[i] < vol_avg[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above pivot or volume drops
            if close[i] > pivot_aligned[i] or volume[i] < vol_avg[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for breakout
            # Volume confirmation: current volume > 1.5x average
            if volume[i] > vol_avg[i] * 1.5:
                # Long breakout: price closes above resistance1
                if close[i] > resistance1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below support1
                elif close[i] < support1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals