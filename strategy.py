#!/usr/bin/env python3
"""
12h_WilliamsAlligator_BullBear
Hypothesis: Williams Alligator identifies market trends and ranging conditions. In trending markets (price outside Alligator mouth), follow the direction with momentum. In ranging markets (price inside mouth), fade extremes. Uses 1-day trend filter and volume confirmation to avoid false signals. Designed for 12h timeframe to target 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: three SMAs (Jaw=13, Teeth=8, Lips=5) with future shift
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align HTF indicators to LTF with proper delay
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions
        # Market is trending when all three lines are ordered and separated
        # Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        # Market is ranging when lines are intertwined
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Check for clear separation (trending market)
        uptrend_alligator = jaw_val > teeth_val and teeth_val > lips_val
        downtrend_alligator = jaw_val < teeth_val and teeth_val < lips_val
        ranging_market = not (uptrend_alligator or downtrend_alligator)
        
        # Price position relative to Alligator
        price_above_alligator = close[i] > max(jaw_val, teeth_val, lips_val)
        price_below_alligator = close[i] < min(jaw_val, teeth_val, lips_val)
        price_inside_mouth = not (price_above_alligator or price_below_alligator)
        
        # Trend filter from 1-day EMA
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic
        # In trending markets: follow Alligator direction with volume
        # In ranging markets: fade extremes (mean reversion)
        long_entry = False
        short_entry = False
        
        if uptrend_alligator and vol_confirm:
            # Strong uptrend: go long on pullbacks to teeth
            if close[i] <= teeth_val and price_below_alligator:
                long_entry = True
        elif downtrend_alligator and vol_confirm:
            # Strong downtrend: go short on retracements to teeth
            if close[i] >= teeth_val and price_above_alligator:
                short_entry = True
        elif ranging_market and vol_confirm:
            # Ranging market: fade extremes
            if price_below_alligator and close[i] > lips_val:  # bounce from lower band
                long_entry = True
            elif price_above_alligator and close[i] < teeth_val:  # reject from upper band
                short_entry = True
        
        # Exit logic: opposite signal or loss of momentum
        long_exit = (price_above_alligator and close[i] < jaw_val) or \
                    (not vol_confirm) or \
                    (downtrend_alligator and close[i] < teeth_val)
        short_exit = (price_below_alligator and close[i] > jaw_val) or \
                     (not vol_confirm) or \
                     (uptrend_alligator and close[i] > teeth_val)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_BullBear"
timeframe = "12h"
leverage = 1.0