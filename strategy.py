#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA trend filter + volume confirmation
# - Primary signal: Williams Alligator (Jaw/Teeth/Lips) alignment on 1d timeframe
#   - Long: Lips > Teeth > Jaw (bullish alignment)
#   - Short: Lips < Teeth < Jaw (bearish alignment)
# - Trend filter: 1w EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 1d volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Alligator identifies trends, EMA50 filter ensures alignment with higher timeframe trend,
#   volume confirmation reduces false signals in ranging markets

name = "1d_1w_alligator_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Williams Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Smoothed Moving Average (SMA) with specific periods
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    # Using SMA as approximation for SmMA (similar for trend identification)
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Apply forward shifts (SmMA characteristic)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set invalid values at the beginning due to shift
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Williams Alligator lines to 1d timeframe (completed 1d bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price crosses below EMA50
            if lips_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= jaw_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price crosses above EMA50
            if lips_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= jaw_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with volume confirmation and EMA50 filter
            # Long: Lips > Teeth > Jaw (bullish alignment) AND volume regime AND price above EMA50
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and volume_regime[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Lips < Teeth < Jaw (bearish alignment) AND volume regime AND price below EMA50
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and volume_regime[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals