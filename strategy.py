#!/usr/bin/env python3
# 6h_1d_weekly_pivot_volume_breakout_v1
# Hypothesis: Breakouts above weekly R4 or below weekly S4 on 6h timeframe with volume confirmation and 1-day trend filter.
# Weekly pivot levels provide strong institutional support/resistance. Breakouts with volume indicate institutional participation.
# In bull markets: buy weekly R4 breakouts with volume and 1d uptrend.
# In bear markets: sell weekly S4 breakdowns with volume and 1d downtrend.
# Weekly pivot calculation uses prior week's high/low/close to avoid look-ahead.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6sma20 for trend filter (short-term trend)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data to avoid look-ahead)
    # Standard pivot: P = (H + L + C) / 3
    # R4 = P + 3*(H - L)  [equivalent to: R4 = C + 3*(H - L)]
    # S4 = P - 3*(H - L)  [equivalent to: S4 = C - 3*(H - L)]
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Weekly pivot and levels
    weekly_p = (prev_high + prev_low + prev_close) / 3
    weekly_r4 = prev_close + 3 * (prev_high - prev_low)  # R4 level
    weekly_s4 = prev_close - 3 * (prev_high - prev_low)  # S4 level
    
    # Align weekly levels to 6h timeframe
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Get 1-day trend filter (50-period SMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S4 or trend reverses (price < SMA20)
            if close[i] < weekly_s4_aligned[i] or close[i] < sma20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R4 or trend reverses (price > SMA20)
            if close[i] > weekly_r4_aligned[i] or close[i] > sma20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly R4 with volume surge and 1d uptrend
            if (close[i] > weekly_r4_aligned[i] and vol_surge and 
                close[i] > sma50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly S4 with volume surge and 1d downtrend
            elif (close[i] < weekly_s4_aligned[i] and vol_surge and 
                  close[i] < sma50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals