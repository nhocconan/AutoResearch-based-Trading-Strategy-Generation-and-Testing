#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 12h trend filter.
# Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with 8,5,3 shifts.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Trend filter: 12h EMA50 > EMA200 for longs, EMA50 < EMA200 for shorts.
# Entry: Alligator aligned (Lips > Teeth > Jaw for long, reverse for short) + Elder Ray power confirms direction + 12h trend agrees.
# Exit: Alligator misalignment or Elder Ray power contradicts.
# Works in trending markets (bull/bear) by using 12h EMA filter to avoid counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 and EMA200 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Williams Alligator on 6h
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    # Apply shifts: Jaw shift 8, Teeth shift 5, Lips shift 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for lookback period
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Elder Ray on 6h (EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        alligator_long = lips_val > teeth_val and teeth_val > jaw_val  # Lips > Teeth > Jaw
        alligator_short = lips_val < teeth_val and teeth_val < jaw_val  # Lips < Teeth < Jaw
        
        # Elder Ray power
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # 12h trend filter
        ema50_val = ema50_12h_aligned[i]
        ema200_val = ema200_12h_aligned[i]
        trend_up = ema50_val > ema200_val
        trend_down = ema50_val < ema200_val
        
        if position == 0:
            # Long: Alligator aligned up + Bull Power positive + 12h trend up
            if alligator_long and bull_val > 0 and trend_up:
                position = 1
                signals[i] = position_size
            # Short: Alligator aligned down + Bear Power negative + 12h trend down
            elif alligator_short and bear_val < 0 and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator misalignment or Bull Power negative or 12h trend down
            if not (alligator_long and bull_val > 0 and trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator misalignment or Bear Power positive or 12h trend up
            if not (alligator_short and bear_val < 0 and trend_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Alligator_ElderRay"
timeframe = "6h"
leverage = 1.0