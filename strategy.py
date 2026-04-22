#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator with 1-day trend filter and volume confirmation.
The Williams Alligator uses smoothed moving averages (Jaw, Teeth, Lips) to identify
trending vs ranging markets. When all three lines are aligned (all above/below price
and in correct order), it indicates a strong trend. The 1-day trend filter ensures
trades align with the daily trend to avoid counter-trend trades. Volume spikes
confirm institutional participation at trend continuation points.
This strategy aims to capture strong trending moves in both bull and bear markets
by trading in the direction of the Alligator alignment with trend and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Smoothing"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Value) / Period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5)"""
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    jaw = smma(median_price, 13)
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    teeth = smma(median_price, 8)
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA, shifted 3 bars ahead
    lips = smma(median_price, 5)
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h Alligator data - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Alligator on 12h data
    jaw_12h, teeth_12h, lips_12h = calculate_alligator(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values
    )
    
    # Align Alligator components to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        # Bullish alignment: Lips > Teeth > Jaw (all above price indicates uptrend)
        # Bearish alignment: Lips < Teeth < Jaw (all below price indicates downtrend)
        bullish_alignment = (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i])
        bearish_alignment = (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i])
        
        if position == 0:
            # Long: bullish alignment, price above all lines, above 1d EMA, volume spike
            if (bullish_alignment and
                close[i] > lips_12h_aligned[i] and
                close[i] > teeth_12h_aligned[i] and
                close[i] > jaw_12h_aligned[i] and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below all lines, below 1d EMA, volume spike
            elif (bearish_alignment and
                  close[i] < lips_12h_aligned[i] and
                  close[i] < teeth_12h_aligned[i] and
                  close[i] < jaw_12h_aligned[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: alignment breaks or price crosses opposite side
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or price below teeth
                if bearish_alignment or close[i] < teeth_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish alignment or price above teeth
                if bullish_alignment or close[i] > teeth_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0