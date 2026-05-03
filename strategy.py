#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend via smoothed medians.
# In 6h timeframe, Alligator convergence signals ranging market (no trade).
# Divergence (jaws, teeth, lips aligned and separated) with 1d EMA34 trend and volume spike
# captures strong trending moves. Designed for 12-37 trades/year on 6h to minimize fee drag.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h close (smoothed medians)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 6h data for volume confirmation (using same timeframe)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: 
        # Bullish alignment: Lips > Teeth > Jaw (all separated and ordered)
        # Bearish alignment: Lips < Teeth < Jaw (all separated and ordered)
        bullish_aligned = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        bearish_aligned = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        # Calculate separation (minimum gap between lines as % of price)
        gap_long = (lips_shifted[i] - jaw_shifted[i]) / close[i] if close[i] > 0 else 0
        gap_short = (jaw_shifted[i] - lips_shifted[i]) / close[i] if close[i] > 0 else 0
        min_separation = 0.005  # 0.5% minimum separation to avoid choppy signals
        
        if position == 0:
            # Long: bullish Alligator alignment + 1d uptrend + volume spike + sufficient separation
            if (bullish_aligned and gap_long > min_separation and 
                ema_34_1d_aligned[i] > close[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + 1d downtrend + volume spike + sufficient separation
            elif (bearish_aligned and gap_short > min_separation and 
                  ema_34_1d_aligned[i] < close[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment or 1d trend turns down
            if not bullish_aligned or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment or 1d trend turns up
            if not bearish_aligned or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals