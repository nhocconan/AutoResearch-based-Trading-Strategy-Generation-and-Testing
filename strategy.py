#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) combination
# Williams Alligator (jaw/teeth/lips) defines trend direction and alignment
# Elder Ray power (bull/bear) from 1d measures trend strength via EMA13
# Long when: Alligator aligned bullish (lips>teeth>jaw) AND 1d bull power > 0
# Short when: Alligator aligned bearish (jaw>teeth>lips) AND 1d bear power < 0
# Uses discrete sizing (0.25) to minimize fee churn. Designed for 6h timeframe
# to capture medium-term trends in both bull and bear markets via trend strength confirmation.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_WilliamsAlligator_1dElderRay_TrendStrength"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 on 1d close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema13_1d  # Bull Power: measures bullish strength
    bear_power_1d = low_1d - ema13_1d   # Bear Power: measures bearish strength (negative when weak)
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams Alligator on 6h: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (simple moving average)
    median_price = (high + low) / 2
    
    # Jaw (13, 8)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    jaw_values = jaw.values
    
    # Teeth (8, 5)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    teeth_values = teeth.values
    
    # Lips (5, 3)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(lips_values[i]) or np.isnan(teeth_values[i]) or np.isnan(jaw_values[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment signals
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_aligned = lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i]
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_aligned = jaw_values[i] > teeth_values[i] and teeth_values[i] > lips_values[i]
        
        # Elder Ray signals from 1d
        # Bull power > 0 indicates bullish strength
        # Bear power < 0 indicates bearish strength (more negative = stronger bearish)
        bull_power_signal = bull_power_aligned[i] > 0
        bear_power_signal = bear_power_aligned[i] < 0
        
        if position == 0:
            # Long: Bullish Alligator alignment AND bullish 1d Elder Ray
            if bullish_aligned and bull_power_signal:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND bearish 1d Elder Ray
            elif bearish_aligned and bear_power_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR bear power turns positive (weakening bull)
            if not bullish_aligned or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR bull power turns negative (weakening bear)
            if not bearish_aligned or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals