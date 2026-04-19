# 6h_WeeklyPivot_Continuation - Weekly Pivot Breakout with Volume Filter
# Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance.
# Breaking above R4 with volume indicates bullish momentum; breaking below S4 indicates bearish momentum.
# Weekly timeframe filters out daily noise, suitable for 6h entries.
# Works in both bull/bear: In bull, buy R4 breakouts; in bear, sell S4 breakdowns.
# Target: 15-25 trades/year per symbol with disciplined entries.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Continuation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly high/low/close for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels: P = (H+L+C)/3, R4 = P + 3*(H-L), S4 = P - 3*(H-L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r4 = pp + 3.0 * (weekly_high - weekly_low)
    s4 = pp - 3.0 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with volume confirmation
            if close[i] > r4_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume confirmation
            elif close[i] < s4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below R4 (failed breakout) or volume drops
            if close[i] < r4_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above S4 (failed breakdown) or volume drops
            if close[i] > s4_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals