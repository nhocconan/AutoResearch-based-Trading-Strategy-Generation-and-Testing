#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 filter and volume spike confirmation.
# Williams Alligator uses SMAs of median price (HL/2) to identify trends.
# Jaw (13-period), Teeth (8-period), Lips (5-period) - all shifted forward.
# When Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend.
# 1d EMA34 filter ensures alignment with daily trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Designed to work in both bull and bear markets by following the trend.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components
    median_price = (high + low) / 2.0
    median_series = pd.Series(median_price)
    
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = median_series.rolling(window=13, min_periods=13).mean().shift(8).values
    
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = median_series.rolling(window=8, min_periods=8).mean().shift(5).values
    
    # Lips (5-period SMA, shifted 3 bars)
    lips = median_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw AND uptrend AND volume spike
            if lips_above_teeth and teeth_above_jaw and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw AND downtrend AND volume spike
            elif lips_below_teeth and teeth_below_jaw and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns bearish OR trend reverses
            if not (lips_above_teeth and teeth_above_jaw) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns bullish OR trend reverses
            if not (lips_below_teeth and teeth_below_jaw) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals