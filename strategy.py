#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume spike
# Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets.
# Enter long when lips > teeth > jaw (bullish alignment) with volume spike and price > 1d EMA34.
# Enter short when lips < teeth < jaw (bearish alignment) with volume spike and price < 1d EMA34.
# Exit when Alligator lines cross (trend weakening) or price crosses 1d EMA34.
# Works in both bull and bear markets by aligning with daily trend and using Alligator for trend strength.
# Target: 50-150 trades over 4 years (12-37/year) on 12h.

name = "12h_WilliamsAlligator_1dEMA34_Volume"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    if len(close) < 13:
        return np.zeros(n)
    
    # SMMA calculation: SMMA(t) = (SMMA(t-1) * (period-1) + close(t)) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines: jaw shifted 8, teeth shifted 5, lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN for invalid periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 2.0x 20-period average
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and 1d EMA)
    start_idx = max(34, 13, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: lips > teeth > jaw
            bullish = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
            # Bearish alignment: lips < teeth < jaw
            bearish = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
            
            # Long entry: bullish alignment with volume spike AND price > 1d EMA34
            if bullish and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment with volume spike AND price < 1d EMA34
            elif bearish and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines cross (teeth < jaw) OR price < 1d EMA34 (trend change)
            if teeth_shifted[i] < jaw_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (teeth > jaw) OR price > 1d EMA34 (trend change)
            if teeth_shifted[i] > jaw_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals