#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Bounce_Strategy
Hypothesis: On daily timeframe, use weekly pivot points (calculated from prior week's high/low/close) as dynamic support/resistance. Price bounces off these levels with volume confirmation provide high-probability entries. Weekly trend filter (via weekly EMA21) ensures alignment with higher timeframe momentum. Designed for low trade frequency (<15/year) to minimize fee dust and work in both bull/bear via mean-reversion at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pp_weekly - low_weekly
    s1_weekly = 2 * pp_weekly - high_weekly
    
    # Align weekly pivot levels to daily timeframe
    pp_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Weekly trend filter: EMA21 on weekly close
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    weekly_uptrend = close_weekly[-1] > ema21_weekly[-1] if len(close_weekly) > 0 else False  # Will be updated in loop
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or
            np.isnan(s1_weekly_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Update weekly trend based on current weekly bar
        # Find corresponding weekly index for current daily bar
        # We approximate: weekly trend is bullish if price > weekly EMA21
        weekly_uptrend_current = close[i] > ema21_weekly_aligned[i]
        
        # Distance to pivot levels (as fraction of price)
        dist_to_s1 = (close[i] - s1_weekly_aligned[i]) / close[i]
        dist_to_r1 = (r1_weekly_aligned[i] - close[i]) / close[i]
        
        # Entry conditions: price near pivot level with volume surge and trend alignment
        near_support = abs(dist_to_s1) < 0.005  # Within 0.5% of S1
        near_resistance = abs(dist_to_r1) < 0.005  # Within 0.5% of R1
        
        long_entry = near_support and volume_surge[i] and weekly_uptrend_current
        short_entry = near_resistance and volume_surge[i] and not weekly_uptrend_current
        
        # Exit when price moves to opposite pivot level
        long_exit = near_resistance and volume_surge[i]
        short_exit = near_support and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Pivot_Bounce_Strategy"
timeframe = "1d"
leverage = 1.0