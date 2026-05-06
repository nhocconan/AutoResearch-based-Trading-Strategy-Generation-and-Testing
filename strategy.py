#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator with Jaw/Teeth/Lips crossover signals
# Long when Alligator Lips cross above Teeth AND price > Jaw AND volume > 1.5 * avg_volume(20)
# Short when Alligator Lips cross below Teeth AND price < Jaw AND volume > 1.5 * avg_volume(20)
# Exit when Lips cross back below Teeth (long) or above Teeth (short)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Williams Alligator identifies trend direction and exhaustion via smoothed SMAs
# Works in bull (Lips above Teeth in uptrend) and bear (Lips below Teeth in downtrend)
# Weekly timeframe provides structural context for daily entries
# Volume confirmation filters weak signals
# Jaw (13-period SMA) acts as dynamic support/resistance

name = "1d_1wWilliamsAlligator_Trend_v1"
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
    
    # Get 1w data ONCE before loop for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:  # Need sufficient data for Alligator (13-period)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator components (all SMAs of median price)
    # Median price = (high + low) / 2
    median_price_1w = (high_1w + low_1w) / 2.0
    
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw_1w = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    jaw_1w = np.roll(jaw_1w, 8)  # shift forward 8 bars
    jaw_1w[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth_1w = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(teeth_1w, 5)  # shift forward 5 bars
    teeth_1w[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips_1w = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(lips_1w, 3)  # shift forward 3 bars
    lips_1w[:3] = np.nan  # first 3 values invalid
    
    # Align 1w Alligator indicators to 1d timeframe (wait for completed 1w bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips cross above Teeth AND price > Jaw AND volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] and lips_aligned[i-1] <= teeth_aligned[i-1] and
                close[i] > jaw_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below Teeth AND price < Jaw AND volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] and lips_aligned[i-1] >= teeth_aligned[i-1] and
                  close[i] < jaw_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips cross back below Teeth
            if lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips cross back above Teeth
            if lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals