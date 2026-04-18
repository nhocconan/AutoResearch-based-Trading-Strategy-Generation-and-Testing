#!/usr/bin/env python3
"""
12h Williams Alligator with 1d Trend Filter
Hypothesis: Williams Alligator identifies market phases (sleeping, awakening, feeding). 
In trending markets (JAWS > TEETH > LIPS for down, reverse for up), we follow the trend.
We use 1d EMA50 as higher timeframe trend filter to ensure alignment with daily momentum.
Entry occurs when Alligator is aligned in trend direction and price crosses TEETH (8-period SMMA).
This strategy aims for 15-25 trades/year to minimize fee decay while capturing sustained moves.
Works in both bull and bear markets by only trading in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=float)
    smma = np.full_like(source, np.nan, dtype=float)
    smma[length-1] = np.mean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components (13,8,5 SMMA of median price)
    median_price = (high + low) / 2.0
    jaws = smma(median_price, 13)  # Blue line (13-period)
    teeth = smma(median_price, 8)   # Red line (8-period)
    lips = smma(median_price, 5)    # Green line (5-period)
    
    # Align Alligator components to LTF
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        trend = ema50_1d_aligned[i]
        
        # Check Alligator alignment: Mouth open in direction of trend
        # Up alignment: Lips > Teeth > Jaws AND price above teeth
        # Down alignment: Lips < Teeth < Jaws AND price below teeth
        if position == 0:
            # Long: Alligator aligned up, price crosses above teeth, uptrend
            if lip > tooth and tooth > jaw and price > tooth and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down, price crosses below teeth, downtrend
            elif lip < tooth and tooth < jaw and price < tooth and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if Alligator alignment breaks or price crosses below teeth
            if not (lip > tooth and tooth > jaw) or price < tooth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if Alligator alignment breaks or price crosses above teeth
            if not (lip < tooth and tooth < jaw) or price > tooth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dTrend"
timeframe = "12h"
leverage = 1.0