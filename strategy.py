#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d volume confirmation
# - Williams Alligator uses SMAs of median price (5,8,13 periods) to identify trends
# - Jaw (13-period): blue line, Teeth (8-period): red line, Lips (5-period): green line
# - Trend direction: when Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend
# - Entries occur when price crosses the Alligator's mouth (Teeth) in trend direction
# - 1d volume > 1.3x 30-period average for confirmation
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in trending markets (both bull/bear) while avoiding whipsaws
# - Target: 15-25 trades/year to minimize fee drag

name = "12h_WilliamsAlligator_1dVolume_v1"
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
    
    # Get 12h median price for Williams Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator lines (SMAs of median price)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (30-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=30, min_periods=30).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (00:00-23:00 UTC for 12h - all hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = np.ones(n, dtype=bool)  # Trade all hours for 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, 30)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.3x 1d average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: Lips > Teeth > Jaw (bullish alignment) + price crosses above Teeth + volume
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > teeth[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: Lips < Teeth < Jaw (bearish alignment) + price crosses below Teeth + volume
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < teeth[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Lips < Teeth (Alligator sleeping - trend weakening)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Lips > Teeth (Alligator sleeping - trend weakening)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals