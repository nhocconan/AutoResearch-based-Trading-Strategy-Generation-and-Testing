#!/usr/bin/env python3
# Hypothesis: 6h ADX + Williams Alligator combo with 1d trend filter
# Long when Alligator jaws turn up (green > red > blue) and ADX > 25 with 1d uptrend
# Short when Alligator jaws turn down (blue > red > green) and ADX > 25 with 1d downtrend
# Exit when Alligator lines re-cross or ADX falls below 20
# Uses Alligator for trend initiation, ADX for trend strength, 1d EMA for higher timeframe filter
# Designed to catch strong trends while avoiding choppy markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_Alligator_ADX_1dEMA50_Trend"
timeframe = "6h"
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
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)   # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)    # Green line
    
    # ADX calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Align Alligator and ADX to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth.values)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips.values)
    adx_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), adx.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]  # Green > Red > Blue
        alligator_short = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]  # Blue > Red > Green
        
        if position == 0:
            # Enter long: Alligator bullish alignment, ADX > 25, 1d uptrend
            if (alligator_long and adx_aligned[i] > 25 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish alignment, ADX > 25, 1d downtrend
            elif (alligator_short and adx_aligned[i] > 25 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator re-crosses or ADX < 20
            if not alligator_long or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator re-crosses or ADX < 20
            if not alligator_short or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals