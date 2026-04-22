#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator with 1-day trend filter and volume confirmation.
The Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips) that
indicate trend direction and strength. When the lines are intertwined, the market is ranging
(sleeping Alligator). When they diverge in order (Lips > Teeth > Jaw for uptrend, reverse for
downtrend), a strong trend is present. We use the 1-day EMA to filter for the dominant trend
direction and volume spikes to confirm institutional participation. This strategy aims to
capture strong trending moves while avoiding choppy markets, suitable for both bull and bear
conditions by trading in the direction of the dominant trend with Alligator confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(array, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if len(array) < period:
        return np.full_like(array, np.nan, dtype=float)
    result = np.full_like(array, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(array[:period])
    # Subsequent values: (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(array)):
        result[i] = (result[i-1] * (period-1) + array[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    jaw = smma(df_12h['close'].values, 13)
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    teeth = smma(df_12h['close'].values, 8)
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars ahead
    lips = smma(df_12h['close'].values, 5)
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    lips[:3] = np.nan
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
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
        
        # Determine Alligator alignment
        # Bullish alignment: Lips > Teeth > Jaw (green alignment)
        # Bearish alignment: Lips < Teeth < Jaw (red alignment)
        is_bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        is_bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: bullish alignment, above 1d EMA, volume spike
            if (is_bullish_alignment and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, below 1d EMA, volume spike
            elif (is_bearish_alignment and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: alignment changes or price crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or price below 1d EMA
                if is_bearish_alignment or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish alignment or price above 1d EMA
                if is_bullish_alignment or close[i] > ema_34_1d_aligned[i]:
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