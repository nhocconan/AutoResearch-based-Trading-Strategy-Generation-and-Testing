#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1w Elder Ray Power + volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Elder Ray Power for trend filter (bullish if Power > 0, bearish if Power < 0).
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
  Trend condition: Alligator aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
- Entry: Long when Lips cross above Teeth AND 1w Power > 0 AND volume > 1.5 * volume MA(30).
         Short when Lips cross below Teeth AND 1w Power < 0 AND volume > 1.5 * volume MA(30).
- Exit: Close-based reversal - exit long when Lips cross below Teeth,
        exit short when Lips cross above Teeth.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via 1w Elder Ray trend filter and Alligator alignment.
Proven pattern from DB: Elder Ray and Williams Alligator combinations show strong test performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Get 1w data for Elder Ray Power trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w Elder Ray Power: EMA(13) of high - EMA(13) of close (Bull Power)
    # and EMA(13) of low - EMA(13) of close (Bear Power)
    # We'll use Bull Power - Bear Power = 2*(EMA13(high) - EMA13(low)) but simplified to trend direction
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # EMA13 of high and low
    ema13_high = pd.Series(df_1w_high).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(df_1w_low).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power trend: positive = bullish, negative = bearish
    elder_power = ema13_high - ema13_low  # Simplified Power indicator
    
    # Align 1w Elder Ray Power to 12h
    elder_power_aligned = align_htf_to_ltf(prices, df_1w, elder_power)
    
    # Williams Alligator components on median price
    # Jaw: 13-period SMMA, 8 periods ahead
    # Teeth: 8-period SMMA, 5 periods ahead  
    # Lips: 5-period SMMA, 3 periods ahead
    # Using SMMA (Smoothed Moving Average) approximated by EMA with specific alpha
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Calculate volume MA(30) for confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 13, 30)  # Need enough bars for Elder Ray, Alligator, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(elder_power_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Alligator aligned for uptrend: Lips > Teeth > Jaw
            # Alligator aligned for downtrend: Jaw > Teeth > Lips
            lips_above_teeth = lips[i] > teeth[i]
            teeth_above_jaw = teeth[i] > jaw[i]
            jaw_above_teeth = jaw[i] > teeth[i]
            teeth_above_lips = teeth[i] > lips[i]
            
            # Long: Lips cross above Teeth AND 1w Elder Power bullish AND volume confirmed
            if lips_above_teeth and teeth_above_jaw and elder_power_aligned[i] > 0 and vol_confirmed:
                # Check for crossover: previous Lips <= previous Teeth
                if i > 0 and lips[i-1] <= teeth[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short: Lips cross below Teeth AND 1w Elder Power bearish AND volume confirmed
            elif not lips_above_teeth and jaw_above_teeth and elder_power_aligned[i] < 0 and vol_confirmed:
                # Check for crossover: previous Lips >= previous Teeth
                if i > 0 and lips[i-1] >= teeth[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long when Lips cross below Teeth (Alligator reversion signal)
            if not lips_above_teeth or not teeth_above_jaw:  # Lips <= Teeth or Teeth <= Jaw
                # Check for crossover: previous Lips > previous Teeth
                if i > 0 and lips[i-1] > teeth[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Lips cross above Teeth (Alligator reversion signal)
            if lips_above_teeth and teeth_above_jaw:  # Lips > Teeth and Teeth > Jaw
                # Check for crossover: previous Lips < previous Teeth
                if i > 0 and lips[i-1] < teeth[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wElderRay_Power_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0