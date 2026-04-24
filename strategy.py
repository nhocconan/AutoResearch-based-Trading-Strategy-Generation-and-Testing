#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d Elder Ray trend filter and 1w volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Elder Ray trend filter (bull/bear power) and 1w for volume confirmation.
- Entry: Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND 1w volume > 1.5x 20-period MA.
         Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND 1w volume > 1.5x 20-period MA.
- Exit: Opposite Alligator alignment OR Elder Ray power crosses zero.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator identifies trend via smoothed medians (Jaw=13, Teeth=8, Lips=5).
- Elder Ray measures bull/bear power relative to EMA13 for trend strength.
- 1w volume filter ensures participation from higher timeframe players.
- Works in bull markets (buy on bullish alignment) and bear markets (sell on bearish alignment).
- Estimated trades: ~100 total over 4 years (~25/year) based on trend persistence with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def alligator(high, low, close):
    """Calculate Williams Alligator lines (Jaw, Teeth, Lips)."""
    median_price = (high + low) / 2.0
    jaw = ema(median_price, 13)  # Blue line
    teeth = ema(median_price, 8)  # Red line
    lips = ema(median_price, 5)   # Green line
    return jaw, teeth, lips

def elder_ray(high, low, close):
    """Calculate Elder Ray Bull Power and Bear Power."""
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Elder Ray for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    bull_power, bear_power = elder_ray(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power, additional_delay_bars=1)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power, additional_delay_bars=1)
    
    # Calculate 1w volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    volume_ma_20 = sma(df_1w['volume'].values, 20)
    volume_ratio = df_1w['volume'].values / (volume_ma_20 + 1e-10)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 6h Alligator for current bar
        jaw_i, teeth_i, lips_i = alligator(high[:i+1], low[:i+1], close[:i+1])
        jaw_i = jaw_i[-1]
        teeth_i = teeth_i[-1]
        lips_i = lips_i[-1]
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR Elder Ray power crosses zero
        if position != 0:
            # Exit long: bearish Alligator alignment OR bull power <= 0
            if position == 1:
                if jaw_i > teeth_i > lips_i or bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment OR bear power >= 0
            elif position == -1:
                if jaw_i < teeth_i < lips_i or bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with Elder Ray confirmation and volume filter
        if position == 0:
            # Long: bullish Alligator alignment AND bull power > 0 AND 1w volume > 1.5x MA
            if jaw_i < teeth_i < lips_i and bull_power_aligned[i] > 0 and volume_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND bear power < 0 AND 1w volume > 1.5x MA
            elif jaw_i > teeth_i > lips_i and bear_power_aligned[i] < 0 and volume_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dElderRay_TrendFilter_1wVolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0