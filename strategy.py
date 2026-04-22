#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Weekly pivots (based on prior week's range) provide institutional support/resistance levels.
# Breakouts above weekly R1 or below weekly S1 with volume confirmation (>1.5x 20-period average)
# capture momentum moves. Trend filter uses price relative to weekly pivot point (PP).
# Designed for low trade frequency (~15-25/year) to minimize fee decay. Works in both bull
# and bear markets by following higher timeframe pivot levels.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pp_1w) - low_1w
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly pivots to 6h timeframe (waits for weekly bar to close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 6h close
    high_series = prices['high']
    low_series = prices['low']
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pp = pp_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        dch = donchian_high[i]
        dcl = donchian_low[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND above weekly R1 with volume
            if price > dch and price > r1 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND below weekly S1 with volume
            elif price < dcl and price < s1 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to weekly pivot point
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to or below weekly PP
                if price <= pp:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to or above weekly PP
                if price >= pp:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0