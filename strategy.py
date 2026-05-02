#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
# When Alligator lines are entangled (close together) = ranging market (avoid trades)
# When Alligator lines are separated (Jaw > Teeth > Lips for uptrend, reverse for downtrend) = trending market
# 1d EMA50 provides higher timeframe trend filter to align with dominant momentum
# Volume spike (2.0x 20-period average) confirms breakout conviction
# Targets 50-150 trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by only trading when Alligator shows clear trend separation

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator from 12h data (Jaw=13, Teeth=8, Lips=5)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMMA of median price, shifted 8 bars)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA of median price, shifted 3 bars)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator trend conditions:
    # Uptrend: Lips > Teeth > Jaw (all lines separated and ascending)
    # Downtrend: Jaw > Teeth > Lips (all lines separated and descending)
    # Ranging: lines are entangled (avoid trading)
    alligator_uptrend = (lips > teeth) & (teeth > jaw)
    alligator_downtrend = (jaw > teeth) & (teeth > lips)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculation)
    start_idx = 16  # max(13+8, 8+5, 5+3) = 21, but using 16 for safety with shifts
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + price > 1d EMA50 + volume spike
            if alligator_uptrend[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + price < 1d EMA50 + volume spike
            elif alligator_downtrend[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator trend changes to downtrend or ranging
            if not alligator_uptrend[i]:  # Exit when uptrend condition breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator trend changes to uptrend or ranging
            if not alligator_downtrend[i]:  # Exit when downtrend condition breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals