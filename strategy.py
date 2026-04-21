#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator crossover with 1d trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) with ADX>25 and volume>1.5x average.
Short when jaws cross below teeth with ADX>25 and volume>1.5x average.
Exit when jaws cross back through teeth or 2x ATR stop.
Alligator uses smoothed moving averages (SMMA) to reduce whipsaws in ranging markets.
Designed for 15-30 trades/year to minimize fee drag while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Alligator components and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator components (SMMA)
    # Jaw: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price (not used for signals)
    median_price = (high_1d + low_1d) / 2
    jaw = smma(median_price, 13)    # Blue line (13-period)
    teeth = smma(median_price, 8)   # Red line (8-period)
    lips = smma(median_price, 5)    # Green line (5-period) - not used
    
    # Calculate 14-period ADX for trend filter
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[1:] = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Alligator components and ADX to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: jaw crosses above teeth with volume and trend
            if (jaw_val > teeth_val and 
                adx_val > 25 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: jaw crosses below teeth with volume and trend
            elif (jaw_val < teeth_val and 
                  adx_val > 25 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: jaw crosses back through teeth OR ATR-based stoploss
            exit_signal = False
            
            # Jaw/teeth crossover exit
            if position == 1 and jaw_val < teeth_val:
                exit_signal = True
            elif position == -1 and jaw_val > teeth_val:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry approximation)
            if position == 1:
                # For longs, stop below teeth (as proxy for entry area)
                if price_close < teeth_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above jaw (as proxy for entry area)
                if price_close > jaw_val + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_JawTeethCross_1dADX25_Volume1.5x_ATR2x"
timeframe = "4h"
leverage = 1.0