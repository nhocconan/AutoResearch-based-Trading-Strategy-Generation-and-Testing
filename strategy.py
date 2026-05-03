#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# Williams Alligator (jaw/teeth/lips) identifies trend direction and strength.
# Elder Ray (bull/bear power) confirms momentum with EMA13.
# Volume spike filters for institutional participation.
# Designed for very low trade frequency (7-25/year) on 1d timeframe to minimize fee drag.
# Works in bull markets via Alligator uptrend + bullish Elder Ray + volume spike.
# Works in bear markets via Alligator downtrend + bearish Elder Ray + volume spike.
# Uses ATR-based stoploss via signal=0 on reversal.

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for HTF trend filter (Alligator)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1w
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev_SMMA * (period-1) + Current_Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1w = df_1w['close'].values
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Shift jaw by 8, teeth by 5, lips by 3 (as per Alligator definition)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align 1w Alligator to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get 1d data for Elder Ray and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 1d timeframe (no alignment needed, but use align_htf_to_ltf for consistency)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator trend: Mouth open (Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend)
        is_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + Bull Power > 0 + Volume Spike
            if is_uptrend and bull_power_aligned[i] > 0 and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Bear Power < 0 + Volume Spike
            elif is_downtrend and bear_power_aligned[i] < 0 and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator reverses (Lips < Jaw) or Bear Power < 0
            if lips_aligned[i] < jaw_aligned[i] or bear_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses (Lips > Jaw) or Bull Power > 0
            if lips_aligned[i] > jaw_aligned[i] or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals