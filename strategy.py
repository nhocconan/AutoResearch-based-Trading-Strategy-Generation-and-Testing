#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1w regime filter
# - Williams Alligator (jaw=13, teeth=8, lips=5) from 1w defines trend direction
# - Elder Ray Power (Bull/Bear) from 1d measures momentum strength
# - Long when: price > Alligator teeth AND Bull Power > 0 AND Bear Power < 0 (1w uptrend)
# - Short when: price < Alligator teeth AND Bear Power > 0 AND Bull Power < 0 (1w downtrend)
# - Uses 1w HTF for regime (avoids whipsaw in sideways markets), 1d for entry timing
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years) to stay within trade limits
# - Works in bull/bear: 1w Alligator adapts to regime, Elder Ray filters weak momentum

name = "6h_1w_1d_alligator_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Elder Ray Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA for Elder Ray (standard)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Pre-compute 1w Williams Alligator (Smoothed Median Price)
    # Typical Price = (H+L+C)/3
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    
    # Alligator Jaw (blue): 13-period SMMA, shifted 8 bars
    jaw_1w = pd.Series(typical_price_1w).rolling(window=13, min_periods=13).mean().values
    jaw_1w = np.roll(jaw_1w, 8)  # shift 8 bars forward
    jaw_1w[:8] = np.nan  # first 8 values invalid
    
    # Alligator Teeth (red): 8-period SMMA, shifted 5 bars
    teeth_1w = pd.Series(typical_price_1w).rolling(window=8, min_periods=8).mean().values
    teeth_1w = np.roll(teeth_1w, 5)  # shift 5 bars forward
    teeth_1w[:5] = np.nan  # first 5 values invalid
    
    # Alligator Lips (green): 5-period SMMA, shifted 3 bars
    lips_1w = pd.Series(typical_price_1w).rolling(window=5, min_periods=5).mean().values
    lips_1w = np.roll(lips_1w, 3)  # shift 3 bars forward
    lips_1w[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator components to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for all indicators
        # Skip if any required data is invalid
        if (np.isnan(teeth_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w regime filter: Alligator alignment
        # Jaw > Teeth > Lips = uptrend (bullish alignment)
        bullish_alligator = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        # Lips > Teeth > Jaw = downtrend (bearish alignment)
        bearish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        
        # 1d Elder Ray conditions
        bullish_elder = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        bearish_elder = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Alligator teeth AND bullish Alligator AND bullish Elder Ray
            if prices['close'].iloc[i] > teeth_aligned[i] and bullish_alligator and bullish_elder:
                position = 1
                signals[i] = 0.25
            # Short conditions: price < Alligator teeth AND bearish Alligator AND bearish Elder Ray
            elif prices['close'].iloc[i] < teeth_aligned[i] and bearish_alligator and bearish_elder:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray divergence or Alligator cross
            exit_long = (bull_power_aligned[i] <= 0) or (lips_aligned[i] > teeth_aligned[i])  # Bull Power weak or lips cross above teeth
            exit_short = (bear_power_aligned[i] <= 0) or (jawaligned[i] < teeth_aligned[i])  # Bear Power weak or jaw cross below teeth
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals