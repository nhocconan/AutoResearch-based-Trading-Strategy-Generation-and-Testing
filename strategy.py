#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trends, with 1d EMA50 for trend alignment.
# Volume > 1.5x 20-period EMA ensures institutional participation. Designed for both bull and bear markets.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.
name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with specific periods
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(data, period):
        sma = np.full(len(data), np.nan)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift jaws/teeth/lips by their respective offsets to avoid look-ahead
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_12h[i] > teeth_12h[i]
        teeth_above_jaw = teeth_12h[i] > jaw_12h[i]
        lips_below_teeth = lips_12h[i] < teeth_12h[i]
        teeth_below_jaw = teeth_12h[i] < jaw_12h[i]
        
        if position == 0:
            # Long: Alligator aligned up + price above lips + volume spike + above 1d EMA50
            if (lips_above_teeth and teeth_above_jaw and price > lips_12h[i] and 
                vol_spike[i] and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + price below lips + volume spike + below 1d EMA50
            elif (lips_below_teeth and teeth_below_jaw and price < lips_12h[i] and 
                  vol_spike[i] and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down (lips < teeth) or price crosses below teeth
            if lips_12h[i] < teeth_12h[i] or price < teeth_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up (lips > teeth) or price crosses above teeth
            if lips_12h[i] > teeth_12h[i] or price > teeth_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals