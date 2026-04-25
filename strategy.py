#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + 1d Fractal confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Williams fractal confirmation (requires 2-bar delay).
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price.
  Trend up: Lips > Teeth > Jaw. Trend down: Lips < Teeth < Jaw.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
  Bullish: Bull Power > 0 and rising. Bearish: Bear Power < 0 and falling.
- Entry: Long when Alligator bullish AND Bull Power > 0 AND bullish 1d fractal.
         Short when Alligator bearish AND Bear Power < 0 AND bearish 1d fractal.
- Exit: Opposite Alligator alignment (trend change).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to catch strong trends with confirmation from multiple time-tested indicators.
- Works in bull markets (catching uptrends) and bear markets (catching downtrends).
- Uses 1d fractals for confirmation to avoid false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h median price for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    median_12h = (high_12h + low_12h) / 2.0
    
    # Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs on median price
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 12h EMA13 for Elder Ray
    close_12h = df_12h['close'].values
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # Calculate 12h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_12h - ema_13_12h
    bear_power = low_12h - ema_13_12h
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate 1d Williams Fractals (requires 2-bar confirmation delay)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5)  # Need 13 for Jaw, 8 for Teeth, 5 for Lips
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_13_12h_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        alligator_bullish = lips_val > teeth_val and teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        # Elder Ray conditions
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Elder Ray rising/falling (1-bar change)
        if i > start_idx:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            bull_power_rising = bull_power_val > bull_power_prev
            bear_power_falling = bear_power_val < bear_power_prev
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        # 1d Fractal conditions
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND rising AND bullish fractal
            long_condition = (
                alligator_bullish and 
                bull_power_val > 0 and 
                bull_power_rising and 
                bullish_fractal_val
            )
            
            # Short: Alligator bearish AND Bear Power < 0 AND falling AND bearish fractal
            short_condition = (
                alligator_bearish and 
                bear_power_val < 0 and 
                bear_power_falling and 
                bearish_fractal_val
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions: Opposite Alligator alignment (trend change)
        elif position != 0:
            # Exit long: Alligator turns bearish
            if position == 1 and alligator_bearish:
                signals[i] = 0.0
                position = 0
            # Exit short: Alligator turns bullish
            elif position == -1 and alligator_bullish:
                signals[i] = 0.0
                position = 0
            # Otherwise maintain position
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_ElderRay_1dFractal_Confirm_v1"
timeframe = "12h"
leverage = 1.0