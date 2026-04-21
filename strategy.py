#!/usr/bin/env python3
"""
1d_1w_Alligator_Momentum_V1
Hypothesis: Use Williams Alligator on 1d (Jaws, Teeth, Lips) to identify trend direction and momentum. 
Go long when Lips > Teeth > Jaws (bullish alignment) and price > Lips; short when Lips < Teeth < Jaws (bearish alignment) and price < Lips.
Use 1w ADX > 25 to filter for trending markets only, avoiding whipsaws in ranging periods.
Exit when Alligator alignment breaks (Lips crosses Teeth) or ADX falls below 20.
Designed for low frequency (10-25 trades/year) to minimize fee drag and work in both bull/bear markets by following higher timeframe trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator: SMAs of median price
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to lower timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14) on 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        
        # ADX trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit when trend weakens
        
        if position == 0:
            # Long: bullish alignment + price above lips + strong trend
            if bullish_alignment and price > lips_aligned[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below lips + strong trend
            elif bearish_alignment and price < lips_aligned[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks or trend weakens
            if not bullish_alignment or price < lips_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks or trend weakens
            if not bearish_alignment or price > lips_aligned[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Alligator_Momentum_V1"
timeframe = "1d"
leverage = 1.0