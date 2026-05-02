#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength.
# Alligator sleeping (Jaw/Teeth/Lips intertwined) = ranging market (avoid).
# Alligator awakening (lines diverging) = trending market (trade in direction).
# 1d EMA50 ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation.
# Designed for 12h timeframe to capture medium-term swings with low trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Calculate 1d Williams Alligator (SMMA of median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Align HTF indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and EMA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping condition: lines are intertwined (market ranging)
        jaw_teeth_diff = abs(jaw_1d_aligned[i] - teeth_1d_aligned[i])
        teeth_lips_diff = abs(teeth_1d_aligned[i] - lips_1d_aligned[i])
        lips_jaw_diff = abs(lips_1d_aligned[i] - jaw_1d_aligned[i])
        max_diff = max(jaw_teeth_diff, teeth_lips_diff, lips_jaw_diff)
        avg_price = (jaw_1d_aligned[i] + teeth_1d_aligned[i] + lips_1d_aligned[i]) / 3.0
        sleeping = max_diff < (0.001 * avg_price)  # 0.1% of average price
        
        if position == 0:  # Flat - look for new entries
            # Only trade when Alligator is awakening (not sleeping)
            if not sleeping:
                # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume spike
                if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                    close[i] > ema_50_1d_aligned[i] and 
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume spike
                elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                      close[i] < ema_50_1d_aligned[i] and 
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid ranging markets
        
        elif position == 1:  # Long position
            # Exit: Alligator lines cross (Lips < Teeth) OR price < 1d EMA50 (trend change)
            if lips_1d_aligned[i] < teeth_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (Lips > Teeth) OR price > 1d EMA50 (trend change)
            if lips_1d_aligned[i] > teeth_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals