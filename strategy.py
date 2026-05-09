#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_WilliamsAlligator_Divergence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_price).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, center=False).mean().shift(3).values
    
    # ADX calculation
    plus_dm = np.where((df_1d['high'].diff() > df_1d['low'].diff().abs()) & (df_1d['high'].diff() > 0), df_1d['high'].diff(), 0)
    minus_dm = np.where((df_1d['low'].diff() > df_1d['high'].diff().abs()) & (df_1d['low'].diff() > 0), df_1d['low'].diff(), 0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14).mean().values
    
    # Align to 6h
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough for Alligator and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or 
            np.isnan(lips_6h[i]) or np.isnan(adx_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_6h[i]
        teeth_val = teeth_6h[i]
        lips_val = lips_6h[i]
        adx_val = adx_6h[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25
            if lips_val > teeth_val > jaw_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25
            elif lips_val < teeth_val < jaw_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips < Teeth (bullish alignment broken)
            if lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips > Teeth (bearish alignment broken)
            if lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals