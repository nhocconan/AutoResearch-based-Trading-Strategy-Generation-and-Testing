# Hypothesis: 4h Williams Alligator + volume confirmation + time-of-day filter
# The Williams Alligator (3 SMAs) identifies trending vs ranging markets.
# In trending markets (jaws open), we trade breakouts in the direction of the trend.
# In ranging markets (jaws closed), we avoid trading to prevent whipsaws.
# This works in both bull and bear markets because it adapts to market regime.
# Time-of-day filter (8-20 UTC) avoids low-liquidity periods.
# Expected trade frequency: ~20-40 trades/year per symbol to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs (Jaw, Teeth, Lips)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) - similar to Wilder's smoothing
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume vs 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Time-of-day filter: 8-20 UTC (avoid low liquidity)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Time filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator signals:
        # Jaws open (trending): Lips > Teeth > Jaw (bullish) OR Lips < Teeth < Jaw (bearish)
        # Jaws closed (ranging): lines intertwined
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Enter long: bullish alignment + volume confirmation
            if bullish_aligned and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + volume confirmation
            elif bearish_aligned and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or volume drops
            if bearish_aligned or vol_val < vol_avg_val * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or volume drops
            if bullish_aligned or vol_val < vol_avg_val * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_WilliamsAlligator_Volume_TimeFilter_v1
# Uses Williams Alligator (3 SMAs) to detect trend vs range
# Enters in direction of trend when jaws are aligned
# Requires volume confirmation above 20-period average
# Time filter: 8-20 UTC to avoid low-liquidity periods
# Exits when trend reverses or volume drops significantly
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_WilliamsAlligator_Volume_TimeFilter_v1"
timeframe = "4h"
leverage = 1.0