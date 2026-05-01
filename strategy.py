#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator regime filter
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trending vs ranging markets
# Strategy: Go long when Bull Power > 0 AND price > Alligator Lips (trending up)
# Go short when Bear Power > 0 AND price < Alligator Lips (trending down)
# Filter: Only trade when Alligator is in trending mode (Jaw > Teeth > Lips for up, reverse for down)
# Uses 1d HTF for Alligator to avoid whipsaws, 6h for Elder Ray entries
# Discrete position sizing 0.25, target 50-150 trades over 4 years (12-37/year)
# Works in bull (catch uptrends) and bear (catch downtrends) via symmetric logic

name = "6h_ElderRay_Alligator_Regime_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for Alligator (HTF trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Alligator uses SMMA (Smoothed Moving Average) which is EMA with alpha=1/(period)
    # Jaw: SMMA(13, 8) -> median price smoothed, 8 bars offset
    # Teeth: SMMA(8, 5) -> median price smoothed, 5 bars offset  
    # Lips: SMMA(5, 3) -> median price smoothed, 3 bars offset
    # Where median price = (high + low) / 2
    
    median_price_1d = (df_1d['high'] + df_1d['low']) / 2
    
    # SMMA approximation using EMA with specific alpha (close enough for strategy)
    # SMMA(t) = SMMA(yesterday) * (N-1)/N + price/N
    # Which is equivalent to EMA with alpha = 1/N
    jaw_1d = pd.Series(median_price_1d).ewm(alpha=1/13, adjust=False).mean().values
    teeth_1d = pd.Series(median_price_1d).ewm(alpha=1/8, adjust=False).mean().values
    lips_1d = pd.Series(median_price_1d).ewm(alpha=1/5, adjust=False).mean().values
    
    # Apply offsets: Jaw shifted 8 bars, Teeth 5 bars, Lips 3 bars
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Fill rolled values with NaN for lookback period
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate Elder Ray on 6h (LTF for entries)
    # Need EMA13 of close for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA13 and Alligator
    start_idx = max(13, 13)  # 13
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13[i]) or 
            np.isnan(jaw_1d_aligned[i]) or
            np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Alligator trend detection on 1d (HTF)
        # Trending up: Jaw > Teeth > Lips (alligator mouth open upward)
        # Trending down: Jaw < Teeth < Lips (alligator mouth open downward)
        # Ranging: otherwise (alligator sleeping)
        trending_up = (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i])
        trending_down = (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i])
        
        # Elder Ray signals on 6h (LTF)
        bullish = bull_power[i] > 0  # Bull Power positive
        bearish = bear_power[i] > 0  # Bear Power positive
        
        # Price relative to Alligator Lips for entry confirmation
        price_above_lips = close[i] > lips_1d_aligned[i]
        price_below_lips = close[i] < lips_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND trending up AND price above lips
            if bullish and trending_up and price_above_lips:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND trending down AND price below lips
            elif bearish and trending_down and price_below_lips:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when bull power turns negative OR alligator turns ranging/down
            if not bullish or not trending_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bear power turns negative OR alligator turns ranging/up
            if not bearish or not trending_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals