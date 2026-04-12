#!/usr/bin/env python3
"""
12h_1d_Williams_Alligator_MeanReversion
Hypothesis: Williams Alligator (3 SMAs) on daily timeframe defines trend (jaws/teeth/lips).
Price crossing lips with 12h momentum confirmation and volume spike triggers mean-reversion trades.
Works in bull/bear: in trend, fade extreme deviations from Alligator; in range, trade reversals at extremes.
Target: 15-30 trades/year via strict Alligator alignment + volume + momentum filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Williams_Alligator_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY WILLIAMS ALLIGATOR ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    # Jaw: 13-period SMMA (smoothed) of median price
    median_price_1d = (high_1d + low_1d) / 2
    jaw = np.full_like(median_price_1d, np.nan)
    if len(median_price_1d) >= 13:
        # SMMA: smoothed moving average
        jaw[12] = np.mean(median_price_1d[:13])
        for i in range(13, len(median_price_1d)):
            jaw[i] = (jaw[i-1] * 12 + median_price_1d[i]) / 13
    
    # Teeth: 8-period SMMA
    teeth = np.full_like(median_price_1d, np.nan)
    if len(median_price_1d) >= 8:
        teeth[7] = np.mean(median_price_1d[:8])
        for i in range(8, len(median_price_1d)):
            teeth[i] = (teeth[i-1] * 7 + median_price_1d[i]) / 8
    
    # Lips: 5-period SMMA
    lips = np.full_like(median_price_1d, np.nan)
    if len(median_price_1d) >= 5:
        lips[4] = np.mean(median_price_1d[:5])
        for i in range(5, len(median_price_1d)):
            lips[i] = (lips[i-1] * 4 + median_price_1d[i]) / 5
    
    # === 12H MOMENTUM (ROC 5) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 6:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    roc_5 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 6:
        roc_5[5:] = (close_12h[5:] - close_12h[:-5]) / close_12h[:-5] * 100
    
    # === VOLUME AVERAGE (20-period) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    roc_5_aligned = align_htf_to_ltf(prices, df_12h, roc_5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(roc_5_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: check if jaws/teeth/lips are ordered (trending) or tangled (ranging)
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Lips < Teeth < Jaw
        # In range: tangled (no clear order)
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        # Alligator is "sleeping" (tangled) when lines are close - ranging market
        alligator_sleeping = (abs(lips_val - teeth_val) < (jaw_val * 0.001) and 
                              abs(teeth_val - jaw_val) < (jaw_val * 0.001))
        
        # Alligator is "awake" (ordered) - trending market
        alligator_awake_up = lips_val > teeth_val > jaw_val
        alligator_awake_down = lips_val < teeth_val < jaw_val
        
        # Volume confirmation: spike > 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # 12h momentum confirmation
        mom_confirm = roc_5_aligned[i] > 0  # positive momentum for long
        mom_confirm_short = roc_5_aligned[i] < 0  # negative momentum for short
        
        # Mean reentry signals: price deviation from lips with confirmation
        # Long: price significantly below lips in ranging market OR 
        #       price below lips with bullish momentum in uptrend
        dev_from_lips = (close[i] - lips_val) / lips_val * 100  # % deviation
        
        long_setup = False
        short_setup = False
        
        if alligator_sleeping:  # ranging market - pure mean reversion
            # Price > 1.5% above lips = short opportunity
            # Price < 1.5% below lips = long opportunity
            long_setup = (dev_from_lips < -1.5) and vol_confirm
            short_setup = (dev_from_lips > 1.5) and vol_confirm
        elif alligator_awake_up:  # uptrend - fade only strong deviations with momentum
            long_setup = (dev_from_lips < -2.0) and vol_confirm and mom_confirm
            short_setup = False  # don't fight uptrend
        elif alligator_awake_down:  # downtrend - fade only strong deviations with momentum
            long_setup = False  # don't fight downtrend
            short_setup = (dev_from_lips > 2.0) and vol_confirm and mom_confirm_short
        
        # Exit when price returns to lips (mean reversion complete) or Alligator wakes up against position
        exit_long = (close[i] >= lips_val) or (position == 1 and alligator_awake_down)
        exit_short = (close[i] <= lips_val) or (position == -1 and alligator_awake_up)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals