#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with regime filter
# - Williams Alligator (Jaws, Teeth, Lips) on 12h defines trend direction (bull/bear/neutral)
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6s measures trend strength
# - Long when: Alligator bullish (Lips > Teeth > Jaws) AND Bull Power > 0 AND Bear Power < 0
# - Short when: Alligator bearish (Lips < Teeth < Jaws) AND Bear Power > 0 AND Bull Power < 0
# - Neutral when: Alligator intertwined (no clear order) - stay flat
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in trending markets (both bull and bear) by following Alligator direction
# - Avoids choppy markets via Alligator intertwined condition

name = "6h_12h_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return signals
    
    # 12h data for Alligator
    close_12h = df_12h['close'].values
    
    # Williams Alligator lines (13,8,5 periods SMAs shifted)
    # Jaws: 13-period SMA shifted 8 bars
    jaws_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6t
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 6h data for Elder Ray (EMA13 of close)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        # Bullish: Lips > Teeth > Jaws (green alignment)
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        # Bearish: Lips < Teeth < Jaws (red alignment)
        alligator_bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        # Neutral: intertwined (no clear order)
        alligator_neutral = not (alligator_bullish or alligator_bearish)
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0  # Bullish momentum
        bear_strong = bear_power[i] > 0  # Bearish momentum
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish AND Bull Power positive AND Bear Power negative
        if alligator_bullish and bull_strong and (bear_power[i] < 0):
            enter_long = True
        
        # Short: Alligator bearish AND Bear power positive AND Bull Power negative
        if alligator_bearish and bear_strong and (bull_power[i] < 0):
            enter_short = True
        
        # Exit conditions: opposite Alligator signal or neutral
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish or neutral
            exit_long = alligator_bearish or alligator_neutral
        elif position == -1:
            # Exit short if Alligator turns bullish or neutral
            exit_short = alligator_bullish or alligator_neutral
        
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