#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND price > 1d EMA50.
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND price < 1d EMA50.
# Exit when Alligator alignment reverses (jaws crosses teeth).
# Uses 6h timeframe with 1d HTF for EMA50 trend filter. Target: 50-150 total trades over 4 years.
# Williams Alligator identifies trend via smoothed medians, Elder Ray measures bull/bear power via EMA13.
# Combines trend direction (Alligator) with momentum strength (Elder Ray) and higher timeframe trend filter.
# Discrete position sizing 0.25 to manage drawdown and reduce fee churn.

name = "6h_Alligator_ElderRay_1dEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three smoothed medians
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    # Teeth (red): 8-period SMMA, shifted 5 bars
    # Lips (green): 5-period SMMA, shifted 3 bars
    # SMMA = Smoothed Moving Average (similar to EMA with alpha=1/period)
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        # First value is SMA, then recursive smoothing
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full(len(values), np.nan)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(values)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + values[i]) / period
        return smma_vals
    
    jaw = smma(high, jaw_period)  # Using high for jaw as per original
    teeth = smma(low, teeth_period)  # Using low for teeth
    lips = smma(close, lips_period)  # Using close for lips
    
    # Apply shifts (Alligator shifts jaws/teeth/lips into future)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    # Set shifted values to NaN for invalid indices
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    elder_period = 13
    ema13 = pd.Series(close).ewm(span=elder_period, adjust=False, min_periods=elder_period).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for all indicators
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, elder_period, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
        bearish_alignment = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        
        if position == 0:
            # LONG: Bullish Alligator alignment + Bull Power > 0 + price > 1d EMA50
            if (bullish_alignment and 
                bull_power[i] > 0 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment + Bear Power < 0 + price < 1d EMA50
            elif (bearish_alignment and 
                  bear_power[i] < 0 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment turns bearish (jaw crosses above teeth)
            if jaw_shifted[i] > teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment turns bullish (jaw crosses below teeth)
            if jaw_shifted[i] < teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals