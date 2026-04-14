#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Weekly Camarilla Pivot Breakout with Volume Confirmation
# Long when price breaks above weekly R4 with volume > 2x 20-period average
# Short when price breaks below weekly S4 with volume > 2x 20-period average
# Exit when price returns to weekly pivot point (PP)
# Weekly Camarilla pivots provide strong support/resistance levels
# Breakouts with volume confirmation capture institutional moves
# Works in both bull/bear markets by trading breakouts in direction of momentum
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Weekly Camarilla Pivot Levels
    # PP = (H + L + C) / 3
    # Range = H - L
    # R4 = PP + Range * 1.5
    # S4 = PP - Range * 1.5
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    r4 = weekly_pivot + weekly_range * 1.5
    s4 = weekly_pivot - weekly_range * 1.5
    pp = weekly_pivot  # pivot point for exit
    
    # Align weekly levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0  # Require 2x average volume for breakout
        
        if position == 0:
            # Long setup: price breaks above weekly R4 with volume confirmation
            if price > r4_aligned[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below weekly S4 with volume confirmation
            elif price < s4_aligned[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot point
            if price <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot point
            if price >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyCamarilla_VolumeBreakout"
timeframe = "6h"
leverage = 1.0