#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation
# Alligator identifies trend direction (JAW, TEETH, LIPS alignment).
# Elder Ray confirms bull/bear power (EMA13 vs high/low).
# Volume spike ensures institutional participation.
# Trend-following with volatility filter works in both bull/bear markets by avoiding range-bound whipsaws.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_ElderRay_Volume"
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
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead
    # Lips: 5-period SMMA, 3 bars ahead
    median_price = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift for Alligator's forward-looking nature
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    
    # Align indicators to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_shifted)
    bull_power_12h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_12h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike detection on 12h (24-period = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_spike = volume > (vol_ma.values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(bull_power_12h[i]) or np.isnan(bear_power_12h[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + Volume spike
            if lips_12h[i] > teeth_12h[i] > jaw_12h[i] and bull_power_12h[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Jaw > Teeth > Lips (bearish alignment) + Bear Power > 0 + Volume spike
            elif jaw_12h[i] > teeth_12h[i] > lips_12h[i] and bear_power_12h[i] > 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR Bear Power negative
            if jaw_12h[i] > teeth_12h[i] > lips_12h[i] or bear_power_12h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR Bull Power negative
            if lips_12h[i] > teeth_12h[i] > jaw_12h[i] or bull_power_12h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals