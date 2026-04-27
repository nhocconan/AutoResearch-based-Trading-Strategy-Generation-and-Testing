#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with Elder Ray power confirmation and 12h trend filter.
# Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Elder Ray (Bull/Bear Power) confirms momentum behind the move.
# Long when Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND 12h close > EMA50.
# Short when Jaw > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND 12h close < EMA50.
# Exit when Alligator alignment breaks or power reverses.
# Designed for ~20-30 trades/year with strong trend confirmation to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Alligator and Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw (13-period, 8 bars ahead), Teeth (8-period, 5 bars ahead), Lips (5-period, 3 bars ahead)
    median_price_12h = (high_12h + low_12h) / 2.0
    
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Get 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough data for slowest indicator (jaw: 13+8=21 bars)
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # Elder Ray power
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # 12h trend filter
        trend_bull = close[i] > ema50_aligned[i]
        trend_bear = close[i] < ema50_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + positive Bull Power + bullish trend
            if bullish_alignment and bull_power > 0 and trend_bull:
                signals[i] = size
                position = 1
            # Short: bearish Alligator alignment + negative Bear Power + bearish trend
            elif bearish_alignment and bear_power < 0 and trend_bear:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not bullish_alignment or bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not bearish_alignment or bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_12hTrend"
timeframe = "4h"
leverage = 1.0