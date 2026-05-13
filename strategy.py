#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d ADX trend filter and volume spike confirmation.
# Long when price > Alligator Jaw with ADX>25 (strong uptrend) and volume > 2x average.
# Short when price < Alligator Jaw with ADX>25 (strong downtrend) and volume > 2x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams Alligator identifies trend direction via smoothed medians (Jaw=13, Teeth=8, Lips=5).
# ADX filter ensures we only trade in trending markets. Volume spike confirms participation.
# Works in bull markets via upward alignment and in bear markets via downward alignment.

name = "12h_WilliamsAlligator_1dADX25_VolumeSpike_v1"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Williams Alligator: Smoothed medians (not moving averages)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        alpha = 1.0 / period
        result = np.zeros_like(arr, dtype=float)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    # Median price for each bar
    median_price = (high + low) / 2.0
    
    # Calculate Alligator lines
    lips = smma(median_price, 5)   # 5-period SMMA
    teeth = smma(median_price, 8)  # 8-period SMMA
    jaw = smma(median_price, 13)   # 13-period SMMA
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # For Alligator, we primarily use Jaw as the trend indicator
    # Lips > Teeth > Jaw = bullish alignment
    # Lips < Teeth < Jaw = bearish alignment
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for Alligator (13+8) and volume MA (20)
    start_idx = max(21, 20)  # 21 to ensure Alligator lines are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) with ADX>25 and volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) with ADX>25 and volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment (Lips < Jaw) OR ADX < 20 (trend weakening)
            if (lips_shifted[i] < jaw_shifted[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment (Lips > Jaw) OR ADX < 20 (trend weakening)
            if (lips_shifted[i] > jaw_shifted[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals