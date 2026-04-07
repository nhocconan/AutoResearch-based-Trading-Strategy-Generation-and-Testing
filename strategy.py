#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Price breaking above/below weekly pivot ranges with volume confirmation
# captures institutional breakout moves. Uses weekly pivot levels (calculated from
# prior week) as support/resistance, volume surge for confirmation, and
# volatility filter to avoid chop. Designed to work in both bull and bear markets
# by capturing genuine breakouts rather than trend following.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_weekly_pivot_range_breakout_volume_v4"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # Resistance1 = 2*Pivot - Low
    # Support1 = 2*Pivot - High
    # Resistance2 = Pivot + (High - Low)
    # Support2 = Pivot - (High - Low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily (using prior week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid choppy markets (ATR ratio < 0.02 indicates low volatility/chop)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ratio = atr / close  # ATR as percentage of price
    vol_filter = atr_ratio > 0.02  # Only trade when volatility is sufficient
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_surge[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below support or volatility drops
            if close[i] < s1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above resistance or volatility drops
            if close[i] > r1_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long breakout: price breaks above resistance with volume surge
            if (close[i] > r1_aligned[i] and vol_surge[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below support with volume surge
            elif (close[i] < s1_aligned[i] and vol_surge[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals