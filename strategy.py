#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Camarilla pivot levels + volume confirmation.
# Uses Camarilla levels calculated from previous day's range to identify mean reversion opportunities.
# Fades at R3/S3 levels (mean reversion) and breaks out at R4/S4 levels (momentum).
# Volume filter confirms institutional participation.
# Designed for 12-30 trades/year to minimize fee drag while capturing both mean reversion and breakout moves.
# Works in bull/bear markets by adapting to volatility regimes.

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate daily range
    daily_range = prev_high - prev_low
    
    # Camarilla levels
    # R4 = close + (high-low) * 1.1/2
    # R3 = close + (high-low) * 1.1/4
    # S3 = close - (high-low) * 1.1/4
    # S4 = close - (high-low) * 1.1/2
    r4 = prev_close + daily_range * 1.1 / 2
    r3 = prev_close + daily_range * 1.1 / 4
    s3 = prev_close - daily_range * 1.1 / 4
    s4 = prev_close - daily_range * 1.1 / 2
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(volume_1d, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_20[:19] = np.nan
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Fade at S3/R3 (mean reversion) - long at S3, short at R3
        fade_long = low[i] <= s3_aligned[i] and vol_filter
        fade_short = high[i] >= r3_aligned[i] and vol_filter
        
        # Breakout at S4/R4 (momentum) - long above R4, short below S4
        breakout_long = high[i] >= r4_aligned[i] and vol_filter
        breakout_short = low[i] <= s4_aligned[i] and vol_filter
        
        # Exit conditions: return to previous day's close
        prev_close_val = prev_close[i] if i < len(prev_close) else np.nan
        if not np.isnan(prev_close_val):
            prev_close_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, prev_close_val))
            exit_long = high[i] >= prev_close_aligned[i] if not np.isnan(prev_close_aligned[i]) else False
            exit_short = low[i] <= prev_close_aligned[i] if not np.isnan(prev_close_aligned[i]) else False
        else:
            exit_long = exit_short = False
        
        # Priority: breakout > fade > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif fade_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (exit_long or (low[i] >= s3_aligned[i] and high[i] <= r3_aligned[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or (low[i] >= s3_aligned[i] and high[i] <= r3_aligned[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals