#!/usr/bin/env python3
# 12h_1d_1w_breakout_pivot_volume
# Hypothesis: Breakouts above daily R4 or below daily S4 on 12h timeframe with volume confirmation and weekly trend filter.
# Daily pivot levels provide strong institutional support/resistance. Breakouts with volume indicate institutional participation.
# In bull markets: buy daily R4 breakouts with volume and weekly uptrend.
# In bear markets: sell daily S4 breakdowns with volume and weekly downtrend.
# Daily pivot calculation uses prior day's high/low/close to avoid look-ahead.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_breakout_pivot_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's data to avoid look-ahead)
    # Standard pivot: P = (H + L + C) / 3
    # R4 = P + 3*(H - L)  [equivalent to: R4 = C + 3*(H - L)]
    # S4 = P - 3*(H - L)  [equivalent to: S4 = C - 3*(H - L)]
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Daily pivot and levels
    daily_p = (prev_high + prev_low + prev_close) / 3
    daily_r4 = prev_close + 3 * (prev_high - prev_low)  # R4 level
    daily_s4 = prev_close - 3 * (prev_high - prev_low)  # S4 level
    
    # Align daily levels to 12h timeframe
    daily_r4_aligned = align_htf_to_ltf(prices, df_1d, daily_r4)
    daily_s4_aligned = align_htf_to_ltf(prices, df_1d, daily_s4)
    
    # Get weekly trend filter (50-period SMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(daily_r4_aligned[i]) or np.isnan(daily_s4_aligned[i]) or
            np.isnan(sma50_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below daily S4 or trend reverses (price < SMA20)
            if close[i] < daily_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above daily R4
            if close[i] > daily_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above daily R4 with volume surge and weekly uptrend
            if (close[i] > daily_r4_aligned[i] and vol_surge and 
                close[i] > sma50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily S4 with volume surge and weekly downtrend
            elif (close[i] < daily_s4_aligned[i] and vol_surge and 
                  close[i] < sma50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals