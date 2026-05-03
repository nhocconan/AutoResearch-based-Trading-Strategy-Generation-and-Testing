#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (JAWS/TEETH/LIPS) with 1d EMA34 trend filter and volume confirmation
# Alligator identifies trendless markets (JAWS/TEETH/LIPS intertwined) vs trending (JAWS > TEETH > LIPS for uptrend, reverse for downtrend).
# Combined with 1d EMA34 to ensure alignment with daily trend and volume spike (1.5x 20 EMA) to filter low-momentum entries.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
# Works in both bull and bear markets by only taking trades in the direction of the 1d EMA34 trend when Alligator confirms trend.

name = "6h_Alligator_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Alligator and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 1d data
    # JAWS (Blue): 13-period SMMA, shifted 8 bars
    # TEETH (Red): 8-period SMMA, shifted 5 bars
    # LIPS (Green): 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    close_1d = df_1d['close'].values
    
    # SMMA calculation using EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        alpha = 1.0 / period
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])  # First value is SMA
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    jaws = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift the lines as per Alligator definition
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that roll in invalid data
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid Alligator and volume EMA
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator trend detection:
        # Uptrend: JAWS > TEETH > LIPS (alligator mouth open upward)
        # Downtrend: JAWS < TEETH < LIPS (alligator mouth open downward)
        # Trendless: lines intertwined (no clear order)
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        
        # Check for clear trend separation
        uptrend_alligator = (jaw > tooth) and (tooth > lip)
        downtrend_alligator = (jaw < tooth) and (tooth < lip)
        
        # Price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + price above 1d EMA34 + volume spike
            if uptrend_alligator and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + price below 1d EMA34 + volume spike
            elif downtrend_alligator and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses uptrend or price crosses below 1d EMA34
            if not uptrend_alligator or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses downtrend or price crosses above 1d EMA34
            if not downtrend_alligator or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals