#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends via smoothed medians
# In bull markets: price above all three lines with Lips > Teeth > Jaw + volume spike + price above 1w EMA50
# In bear markets: price below all three lines with Lips < Teeth < Jaw + volume spike + price below 1w EMA50
# Works in both regimes by using Alligator alignment for trend and volume for confirmation
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

name = "1d_Williams_Alligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d timeframe (using median price)
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw.values)  # 1d data, no HTF conversion needed
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, prices, lips.values)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1w trend filter
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Long: bullish alignment + volume spike + price above 1w EMA50
        # Short: bearish alignment + volume spike + price below 1w EMA50
        if position == 0:
            if (bullish_alignment and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (bearish_alignment and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price below 1w EMA50
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price above 1w EMA50
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals