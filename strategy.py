#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volatility filter and volume confirmation.
Longs when price breaks above Donchian upper (20) with 1d ATR ratio > 1.2 and volume > 1.3x average;
Shorts when price breaks below Donchian lower (20) with same conditions.
Exit on price crossing back through Donchian middle (10-period average) or 2x ATR stop.
Designed for 25-40 trades/year to minimize fee rust while capturing volatility breakouts.
Works in both bull (breakouts continue) and bear (false breakouts filtered by volatility).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Current 14-period ATR ratio (today's ATR / 20-period average)
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period high/low, 10-period middle for exit)
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    middle_10 = (pd.Series(prices['high'].values).rolling(window=10, min_periods=10).max().values +
                 pd.Series(prices['low'].values).rolling(window=10, min_periods=10).min().values) / 2
    
    # Volume confirmation: volume spike > 1.3x 20-period average
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
    
    for i in range(40, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(middle_10[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = high_20[i]
        lower = low_20[i]
        middle = middle_10[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above upper with volatility expansion and volume
            if (price_high > upper and 
                atr_ratio_val > 1.2 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower with volatility expansion and volume
            elif (price_low < lower and 
                  atr_ratio_val > 1.2 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: middle line cross OR ATR-based stoploss
            exit_signal = False
            
            # Middle line exit
            if position == 1 and price_close < middle:
                exit_signal = True
            elif position == -1 and price_close > middle:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry level)
            if position == 1:
                # For longs, stop below entry area (lower band as reference)
                if price_close < lower - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above entry area (upper band as reference)
                if price_close > upper + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Vol1.3x_ATRratio1.2x_Vol1.3x_ATR2x"
timeframe = "4h"
leverage = 1.0