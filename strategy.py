#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + volume confirmation + weekly trend filter
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and strength.
# Volume confirms trend strength, and weekly EMA filter ensures alignment with higher timeframe trend.
# Designed to work in both bull and bear markets by only taking trades in the direction of the weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Williams Alligator (13,8,5) on 1d
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw_13 = np.roll(jaw_13, 8)
    jaw_13[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth_8 = np.roll(teeth_8, 5)
    teeth_8[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips (5-period SMMA, shifted 3 bars)
    lips_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips_5 = np.roll(lips_5, 3)
    lips_5[:3] = np.nan  # First 3 values invalid due to shift
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    jaw_13_aligned = align_htf_to_ltf(prices, df_1d, jaw_13)
    teeth_8_aligned = align_htf_to_ltf(prices, df_1d, teeth_8)
    lips_5_aligned = align_htf_to_ltf(prices, df_1d, lips_5)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_13_aligned[i]) or np.isnan(teeth_8_aligned[i]) or
            np.isnan(lips_5_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (alligator waking up, bullish)
        bullish = (lips_5_aligned[i] > teeth_8_aligned[i] > jaw_13_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw (alligator waking up, bearish)
        bearish = (lips_5_aligned[i] < teeth_8_aligned[i] < jaw_13_aligned[i])
        
        # Long entry: bullish alignment + volume spike + price above weekly EMA50
        if bullish and (volume[i] > 1.5 * vol_avg_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and (position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + volume spike + price below weekly EMA50
        elif bearish and (volume[i] > 1.5 * vol_avg_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and (position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposing alignment or loss of momentum
        elif position == 1 and bearish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0