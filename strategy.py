#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams Alligator combination
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
# - Long: Bull Power > 0 AND Bear Power < 0 AND Lips > Teeth > Jaw (bullish alignment) AND volume > 1.2x 20-period average
# - Short: Bull Power < 0 AND Bear Power > 0 AND Lips < Teeth < Jaw (bearish alignment) AND volume > 1.2x 20-period average
# - Exit: Opposite Elder Ray signal (Bull Power and Bear Power both negative for long exit, both positive for short exit)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear power relative to EMA13, Alligator shows trend alignment
# - Works in both bull (strong bull power + bullish alignment) and bear (strong bear power + bearish alignment) markets
# - Volume confirmation filters weak signals

name = "6h_1d_elder_ray_alligator_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return signals
    
    # Pre-compute 1d Elder Ray components
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Pre-compute 1d Williams Alligator components
    jaw_1d = pd.Series(ema13_1d).ewm(span=13, adjust=False, min_periods=13).mean().values  # EMA13, then shift 8
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values    # EMA8, then shift 5
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values    # EMA5, then shift 3
    
    # Apply shifts for Alligator lines
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Fill rolled values with NaN for proper alignment
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Williams Alligator alignment
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips > teeth and teeth > jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips < teeth and teeth < jaw
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) 
        #       AND bullish Alligator alignment AND volume confirmation
        if bull_power > 0 and bear_power < 0 and bullish_alignment and vol_confirm:
            enter_long = True
        
        # Short: Bull Power < 0 (bulls weak) AND Bear Power > 0 (bears in control)
        #        AND bearish Alligator alignment AND volume confirmation
        if bull_power < 0 and bear_power > 0 and bearish_alignment and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when bull power turns negative OR bear power turns positive (loss of bullish control)
            exit_long = bull_power <= 0 or bear_power >= 0
        elif position == -1:
            # Exit short when bull power turns positive OR bear power turns negative (loss of bearish control)
            exit_short = bull_power >= 0 or bear_power <= 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals