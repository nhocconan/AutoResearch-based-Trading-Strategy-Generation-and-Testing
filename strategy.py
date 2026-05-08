# 12h Williams Alligator + Volume Spike + Chop Regime
# Williams Alligator: 13/8/5 SMAs with 8/5/3 shifts. Trend when jaws aligned and price outside mouth.
# Chop filter: Choppiness Index > 61.8 for ranging (fade extremes), < 38.2 for trending (follow breakout).
# Volume: 2x average volume for confirmation.
# Timeframe: 12h (lower frequency to reduce trade count and fee drag).
# Target: 50-150 trades over 4 years, works in both bull (trend follow) and bear (mean revert in chop).
# Uses 1d HTF for trend alignment and chop regime.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WilliamsAlligator_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs with shifts
    # Jaw: 13-period SMA, 8 bars shift
    # Teeth: 8-period SMA, 5 bars shift
    # Lips: 5-period SMA, 3 bars shift
    close_1d = df_1d['close'].values
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    jaw = np.roll(sma_13, 8)  # shift 8 bars forward
    teeth = np.roll(sma_8, 5)  # shift 5 bars forward
    lips = np.roll(sma_5, 3)   # shift 3 bars forward
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Choppiness Index on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    chop = 100 * np.log10(sum_atr / np.where(range_hl > 0, range_hl, 1e-10)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align Chop to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any NaN values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: chop > 61.8 = ranging, chop < 38.2 = trending
            is_ranging = chop_aligned[i] > 61.8
            is_trending = chop_aligned[i] < 38.2
            
            # Alligator alignment: all three lines in order
            jaw_val = jaw_aligned[i]
            teeth_val = teeth_aligned[i]
            lips_val = lips_aligned[i]
            
            # Long conditions
            if is_trending:
                # In trending regime: follow breakout direction
                # Bullish alignment: lips > teeth > jaw and price > lips
                if lips_val > teeth_val > jaw_val and close[i] > lips_val:
                    if vol_ratio[i] > 2.0:
                        signals[i] = 0.25
                        position = 1
            else:  # ranging or neutral
                # In ranging regime: fade extremes, buy when price near lips (support)
                if close[i] < lips_val * 1.02 and close[i] > lips_val * 0.98:
                    if vol_ratio[i] > 2.0:
                        signals[i] = 0.25
                        position = 1
            
            # Short conditions
            if is_trending:
                # Bearish alignment: lips < teeth < jaw and price < lips
                if lips_val < teeth_val < jaw_val and close[i] < lips_val:
                    if vol_ratio[i] > 2.0:
                        signals[i] = -0.25
                        position = -1
            else:  # ranging or neutral
                # In ranging regime: sell when price near lips (resistance)
                if close[i] > lips_val * 0.98 and close[i] < lips_val * 1.02:
                    if vol_ratio[i] > 2.0:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit long: Alligator chaos (lines intertwined) or regime shift to extreme chop
            # Chaos: jaws, teeth, lips not in perfect order
            jaw_val = jaw_aligned[i]
            teeth_val = teeth_aligned[i]
            lips_val = lips_aligned[i]
            is_chaos = not ((lips_val > teeth_val > jaw_val) or (lips_val < teeth_val < jaw_val))
            
            if is_chaos or chop_aligned[i] > 70:  # extreme chop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: same conditions
            jaw_val = jaw_aligned[i]
            teeth_val = teeth_aligned[i]
            lips_val = lips_aligned[i]
            is_chaos = not ((lips_val > teeth_val > jaw_val) or (lips_val < teeth_val < jaw_val))
            
            if is_chaos or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals