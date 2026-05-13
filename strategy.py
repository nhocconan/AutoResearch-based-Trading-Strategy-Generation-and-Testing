#!/usr/bin/env python3
"""
4h_WilliamsAlligator_12hTrend_Volume
Hypothesis: Williams Alligator (Jaw=13sma(8), Teeth=8sma(5), Lips=5sma(3)) on 4h defines market structure. 
Trend alignment: price above all three lines = bullish, below all three = bearish. 
Entry: price closes back inside the Alligator's mouth (between Teeth and Lips) after trending outside, 
with 12h EMA50 trend confirmation and volume > 1.5x 20-period average. 
Exit: price closes outside the Alligator in opposite direction or volume drops below average.
Works in both bull (buy dips) and bear (sell rallies) by fading overextensions toward the mean (Alligator's teeth).
Targets 20-40 trades/year on 4h to avoid fee drag.
"""

name = "4h_WilliamsAlligator_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA of median price, shifted 8 bars)
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    # Lips (5-period SMMA of median price, shifted 3 bars)
    # Using SMA as approximation for SMMA (Smoothed Moving Average)
    median_price = (high + low) / 2
    
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values that don't have enough history
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe (no extra delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator conditions: price relationship to lines
        price_above_all = close[i] > jaw_aligned[i] and close[i] > teeth_aligned[i] and close[i] > lips_aligned[i]
        price_below_all = close[i] < jaw_aligned[i] and close[i] < teeth_aligned[i] and close[i] < lips_aligned[i]
        price_between_teeth_lips = (close[i] > teeth_aligned[i] and close[i] < lips_aligned[i]) or \
                                   (close[i] < teeth_aligned[i] and close[i] > lips_aligned[i])
        
        if position == 0:
            # LONG: price was below all (bearish), now between Teeth and Lips (reverting to mean)
            # with 12h uptrend and volume confirmation
            if (not price_above_all and not price_below_all and price_between_teeth_lips and
                close[i] > ema_50_12h_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # SHORT: price was above all (bullish), now between Teeth and Lips (reverting to mean)
            # with 12h downtrend and volume confirmation
            elif (not price_above_all and not price_below_all and price_between_teeth_lips and
                  close[i] < ema_50_12h_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes above all (renewed uptrend) or below all (strong downtrend)
            if price_above_all or price_below_all:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes below all (renewed downtrend) or above all (strong uptrend)
            if price_below_all or price_above_all:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals