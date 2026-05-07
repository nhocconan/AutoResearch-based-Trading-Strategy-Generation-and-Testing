#!/usr/bin/env python3
# 6h_ADX_WilliamsAlligator_Trend_Follow
# Hypothesis: Combining ADX (trend strength) with Williams Alligator (trend direction) on 6H timeframe.
# ADX > 25 indicates strong trend, while Alligator jaws/teeth/lips alignment determines direction.
# Uses 1D timeframe for trend confirmation to avoid whipsaws. Works in both bull and bear markets
# by capturing strong trends while avoiding sideways markets. Target: 20-40 trades per year.

name = "6h_ADX_WilliamsAlligator_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1D data ONCE for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1D: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaws = median_price.rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5).values    # Red line
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Align Alligator lines to 6H
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ADX calculation on 6H (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough periods for ADX and Alligator
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals: 
        # Bullish alignment: Lips > Teeth > Jaws (green > red > blue)
        # Bearish alignment: Lips < Teeth < Jaws (green < red < blue)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Enter long: bullish alignment + strong trend
            if bullish_alignment and strong_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + strong trend
            elif bearish_alignment and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or weak trend
            if bearish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or weak trend
            if bullish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals