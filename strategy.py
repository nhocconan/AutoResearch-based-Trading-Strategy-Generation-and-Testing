#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume spike
# Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to detect trends.
# In trending markets: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear)
# In ranging markets: lines intertwine. Volume spike filters weak moves.
# Works in both bull/bear by following the trend direction defined by Alligator alignment.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d median price for Alligator calculation
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    median_price = pd.Series(median_price_1d)
    
    # Alligator lines (SMMA = smoothed moving average)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = median_price.rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = median_price.rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = median_price.rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    # Convert to numpy arrays and align to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long conditions: Bullish alignment AND price above EMA50 (uptrend) + volume
        if bullish_alignment and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short conditions: Bearish alignment AND price below EMA50 (downtrend) + volume
        elif bearish_alignment and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0