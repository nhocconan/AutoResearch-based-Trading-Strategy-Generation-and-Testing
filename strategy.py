#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
Longs when price breaks above 4h Donchian high with 1d ATR < 1.5x its 50-period mean (low volatility regime) and volume > 1.5x 20-period average.
Shorts when price breaks below 4h Donchian low under same conditions.
Exit on opposite Donchian band touch or 2x ATR stop.
Designed for 15-35 trades/year to minimize fee drift while capturing explosive moves in low-volatility environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data ONCE for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period mean of ATR for volatility regime filter
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50  # < 1.5 = low volatility regime
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period on 4h)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above Donchian high in low vol regime with volume
            if (price_high > upper and 
                atr_ratio_val < 1.5 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low under same conditions
            elif (price_low < lower and 
                  atr_ratio_val < 1.5 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite Donchian band touch OR ATR-based stoploss
            exit_signal = False
            
            # Opposite band exit
            if position == 1 and price_low < lower:
                exit_signal = True
            elif position == -1 and price_high > upper:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry band)
            if position == 1:
                # For longs, stop below lower band minus 2x ATR
                if price_close < lower - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper band plus 2x ATR
                if price_close > upper + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_ATRFilter_Volume1.5x"
timeframe = "4h"
leverage = 1.0