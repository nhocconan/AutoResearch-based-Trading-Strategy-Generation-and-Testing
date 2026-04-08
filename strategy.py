#!/usr/bin/env python3
# 6d_donchian_weekly_pivot_volume_v1
# Hypothesis: 6-hour Donchian(20) breakouts filtered by weekly pivot direction and volume confirmation.
# In bull markets: Buy breakouts above weekly pivot resistance with volume.
# In bear markets: Sell breakouts below weekly pivot support with volume.
# Weekly pivot provides institutional reference points; volume confirms participation.
# Target: 15-30 trades/year via tight Donchian breakout + pivot + volume confluence.

name = "6d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_len = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i - donchian_len + 1:i + 1])
        lower[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # Weekly pivot points (using weekly OHLC)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Standard pivot: P = (H + L + C)/3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance levels: R1 = 2*P - L, R2 = P + (H - L)
    r1 = 2 * pivot - weekly_low
    r2 = pivot + (weekly_high - weekly_low)
    # Support levels: S1 = 2*P - H, S2 = P - (H - L)
    s1 = 2 * pivot - weekly_high
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(donchian_len, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below weekly S1 or Donchian lower
            if close[i] < s1_aligned[i] or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above weekly R1 or Donchian upper
            if close[i] > r1_aligned[i] or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian breakout above upper with volume and above weekly pivot
            if (close[i] > upper[i] and 
                close[i] > pivot_aligned[i] and 
                volume_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakdown below lower with volume and below weekly pivot
            elif (close[i] < lower[i] and 
                  close[i] < pivot_aligned[i] and 
                  volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals