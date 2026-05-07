#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily timeframe strategy using weekly pivot points (Camarilla) with volume confirmation.
# Long when price breaks above weekly R4 with volume > 1.5x average volume.
# Short when price breaks below weekly S4 with volume > 1.5x average volume.
# Exit when price returns to weekly pivot (PP) level.
# Uses weekly timeframe for structural levels and daily for execution to reduce whipsaw.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets via upside breakouts, in bear markets via downside breakdowns.
# Volume filter ensures only significant breaks are traded, avoiding false signals.
name = "1d_Camarilla_Weekly_Pivot_Volume_Breakout"
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
    
    # Get weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot point (PP)
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3
    range_weekly = high_weekly - low_weekly
    
    # Camarilla levels: R4 = PP + 1.5 * range, S4 = PP - 1.5 * range
    r4_weekly = pp_weekly + 1.5 * range_weekly
    s4_weekly = pp_weekly - 1.5 * range_weekly
    
    # Align weekly levels to daily timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4_weekly)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4_weekly)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: price breaks above weekly R4 with volume confirmation
            long_breakout = close[i] > r4_aligned[i]
            volume_confirm = volume[i] > 1.5 * avg_volume[i]
            
            # Short condition: price breaks below weekly S4 with volume confirmation
            short_breakdown = close[i] < s4_aligned[i]
            volume_confirm_short = volume[i] > 1.5 * avg_volume[i]
            
            if long_breakout and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif short_breakdown and volume_confirm_short:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to or below weekly pivot
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to or above weekly pivot
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals