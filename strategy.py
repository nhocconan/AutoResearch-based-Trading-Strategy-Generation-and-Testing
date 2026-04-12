#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Daily chart strategy using weekly Camarilla levels for trend direction and daily price action for entries.
# Uses weekly Camarilla H4/L4 as trend filter: only long when price above weekly H4, short when below weekly L4.
# Daily entries triggered by price crossing daily Camarilla H3/L3 levels with volume confirmation.
# Weekly timeframe reduces noise and avoids false signals in choppy markets.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets via breakouts above weekly resistance and in bear markets via breakdowns below weekly support.
name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_prev_w = df_1w['high'].shift(1).values
    low_prev_w = df_1w['low'].shift(1).values
    close_prev_w = df_1w['close'].shift(1).values
    
    range_prev_w = high_prev_w - low_prev_w
    # Weekly H4 and L4 levels (strong support/resistance)
    weekly_h4 = close_prev_w + range_prev_w * 1.1 / 2
    weekly_l4 = close_prev_w - range_prev_w * 1.1 / 2
    
    # Align weekly levels to daily timeframe
    weekly_h4_aligned = align_htf_to_ltf(prices, df_1w, weekly_h4)
    weekly_l4_aligned = align_htf_to_ltf(prices, df_1w, weekly_l4)
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day (for entry timing)
    high_prev_d = df_1d['high'].shift(1).values
    low_prev_d = df_1d['low'].shift(1).values
    close_prev_d = df_1d['close'].shift(1).values
    
    range_prev_d = high_prev_d - low_prev_d
    # Daily H3 and L3 levels (entry triggers)
    daily_h3 = close_prev_d + range_prev_d * 1.1 / 4
    daily_l3 = close_prev_d - range_prev_d * 1.1 / 4
    
    # Align daily levels to daily timeframe (already aligned, but keep for consistency)
    daily_h3_aligned = align_htf_to_ltf(prices, df_1d, daily_h3)
    daily_l3_aligned = align_htf_to_ltf(prices, df_1d, daily_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if weekly levels not ready
        if np.isnan(weekly_h4_aligned[i]) or np.isnan(weekly_l4_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if daily levels not ready
        if np.isnan(daily_h3_aligned[i]) or np.isnan(daily_l3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price above weekly H4 (uptrend) and breaks above daily H3 with volume
        if close[i] > weekly_h4_aligned[i] and close[i] > daily_h3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below weekly L4 (downtrend) and breaks below daily L3 with volume
        elif close[i] < weekly_l4_aligned[i] and close[i] < daily_l3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite daily level break
        elif close[i] < daily_l3_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > daily_h3_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals