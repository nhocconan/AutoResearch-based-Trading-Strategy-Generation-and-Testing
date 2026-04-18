#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 filter and volume spike confirmation.
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw
# 1d EMA34 filter ensures we trade only in the direction of the daily trend.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (bullish alignment with uptrend) and bear markets (bearish alignment with downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    sma = series.rolling(window=period, min_periods=period).mean()
    smma_vals = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        if i < period:
            continue
        if i == period:
            smma_vals[i] = sma[i]
        else:
            smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components
    close_s = pd.Series(close)
    jaw = smma(close_s, 13)  # Jaw: 13-period SMMA
    teeth = smma(close_s, 8)  # Teeth: 8-period SMMA
    lips = smma(close_s, 5)   # Lips: 5-period SMMA
    
    # Shift components as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set initial shifted values to NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        if position == 0:
            # Long: bullish Alligator alignment AND uptrend AND volume spike
            if bullish_alignment and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND downtrend AND volume spike
            elif bearish_alignment and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment turns bearish OR trend reverses
            if not bullish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment turns bullish OR trend reverses
            if not bearish_alignment or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals