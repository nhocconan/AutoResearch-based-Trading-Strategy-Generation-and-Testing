#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Elder Ray trend regime and Alligator confirmation.
- Entry: Long when Alligator is bullish (JAW > TEETH > LIPS) AND Bull Power > 0 AND Bear Power < 0.
         Short when Alligator is bearish (JAW < TEETH < LIPS) AND Bear Power < 0 AND Bull Power > 0.
- Exit: Opposite Alligator alignment OR Elder Ray regime shift.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend via smoothed medians (JAW=13, TEETH=8, LIPS=5).
- Elder Ray measures bull/bear power relative to EMA13 to confirm trend strength.
- 1d regime filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
- Works in bull markets (catch uptrends via Alligator) and bear markets (catch downtrends via Alligator).
- Estimated trades: ~100 total over 4 years (~25/year) based on trend persistence with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def smma(values, period):
    """Calculate Smoothed Moving Average (used in Alligator)."""
    # SMMA is similar to EMA but with alpha = 1/period
    return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def alligator(median_price, jaw_period=13, teeth_period=8, lips_period=5,
              jaw_shift=8, teeth_shift=5, lips_shift=3):
    """Calculate Williams Alligator lines."""
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (forward shift = lookback in array terms)
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Fill shifted values with NaN
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    return jaw, teeth, lips

def elder_ray(high, low, close, ema_period=13):
    """Calculate Elder Ray Bull Power and Bear Power."""
    ema_val = ema(close, ema_period)
    bull_power = high - ema_val
    bear_power = low - ema_val
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate median price for Alligator (typical price)
    median_price = (high + low + close) / 3
    
    # Calculate 1d Elder Ray for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    bull_power_1d, bear_power_1d = elder_ray(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d, additional_delay_bars=1)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d, additional_delay_bars=1)
    
    # Calculate 6h Alligator
    jaw, teeth, lips = alligator(median_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions: opposite Alligator alignment OR Elder Ray regime shift
        if position != 0:
            # Exit long: Alligator turns bearish OR Bull Power becomes negative AND Bear Power positive
            if position == 1:
                if not (jaw[i] > teeth[i] > lips[i]) or (bull_power_1d_aligned[i] < 0 and bear_power_1d_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR Bear Power becomes positive AND Bull Power negative
            elif position == -1:
                if not (jaw[i] < teeth[i] < lips[i]) or (bear_power_1d_aligned[i] > 0 and bull_power_1d_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with Elder Ray confirmation
        if position == 0:
            # Bullish Alligator: JAW > TEETH > LIPS
            bullish_alligator = jaw[i] > teeth[i] > lips[i]
            # Bearish Alligator: JAW < TEETH < LIPS
            bearish_alligator = jaw[i] < teeth[i] < lips[i]
            
            # Long: Bullish Alligator AND Bull Power > 0 AND Bear Power < 0
            if bullish_alligator and bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < 0:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND Bear Power < 0 AND Bull Power > 0
            elif bearish_alligator and bear_power_1d_aligned[i] < 0 and bull_power_1d_aligned[i] > 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_1dRegimeFilter_v1"
timeframe = "6h"
leverage = 1.0