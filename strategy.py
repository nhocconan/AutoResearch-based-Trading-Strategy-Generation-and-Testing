#!/usr/bin/env python3
"""
12h_1d_Williams_Alligator_Trend_v1
Hypothesis: On 12h timeframe, trade in direction of Williams Alligator (three SMAs) with 1d trend filter.
Long when price > Alligator Mouth (13-period SMA) and 1d close > 1d EMA50.
Short when price < Alligator Lips (34-period SMA) and 1d close < 1d EMA50.
Exit when price crosses back inside Alligator jaws or 1d trend reverses.
Designed for low trade frequency (15-30 trades/year) by requiring strong trend alignment.
Works in bull markets via long trends and bear markets via short trends.
Williams Alligator is effective in trending markets and avoids whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Williams_Alligator_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (13, 8, 5 SMAs shifted forward)
    # Jaw: 13-period SMA shifted by 8 bars
    # Teeth: 8-period SMA shifted by 5 bars
    # Lips: 5-period SMA shifted by 3 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Alligator values (no shift for alignment - we'll handle via alignment function)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Load 1d data ONCE for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)  # Using 1d as reference for simplicity
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend conditions
        price_above_jaw = close[i] > jaw_aligned[i]
        price_below_lips = close[i] < lips_aligned[i]
        price_between_teeth = (close[i] > teeth_aligned[i]) and (close[i] < jaw_aligned[i])  # In mouth
        
        # 1d trend filter
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: strong trend alignment
        long_entry = price_above_jaw and above_ema and not price_between_teeth
        short_entry = price_below_lips and below_ema and not price_between_teeth
        
        # Exit conditions: price returns to Alligator mouth or trend reversal
        long_exit = price_between_teeth or below_ema
        short_exit = price_between_teeth or above_ema
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals