#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume spike
# Williams Alligator identifies trend presence via jaw-teeth-lips alignment.
# Long when lips > teeth > jaw (bullish alignment), short when lips < teeth < jaw (bearish alignment).
# Uses 1d EMA34 for higher timeframe trend confirmation and volume spike for momentum confirmation.
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
# Works in both bull and bear markets by trading in the direction of the Alligator's alignment.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for EMA34 and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Typically uses median price, but high/low works for simplicity
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Align Alligator lines (already on 12h, no need for HTF alignment)
    # But we need to shift them forward as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment
        is_bullish_alignment = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        is_bearish_alignment = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        if position == 0:
            # Long: bullish Alligator alignment with volume spike and price above EMA34
            if is_bullish_alignment and volume_spike_aligned[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment with volume spike and price below EMA34
            elif is_bearish_alignment and volume_spike_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks bearish or price crosses below EMA34
            if not is_bullish_alignment or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks bullish or price crosses above EMA34
            if not is_bearish_alignment or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals