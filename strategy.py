#!/usr/bin/env python3
"""
Hypothesis: 4h EUR-style daily pivot (PP, R1, S1) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above R1 with volume > 1.3x average and ADX > 20.
Short when price breaks below S1 with volume > 1.3x average and ADX > 20.
Exit when price returns to pivot point (PP) or volume drops below average.
Pivot levels are widely watched, effective in both trending and ranging markets.
Designed for ~25-35 trades/year to minimize fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close for pivots and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Calculate ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align pivots and ADX to 4h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above R1, volume surge, ADX > 20
            if (price_close > r1_aligned[i] and 
                vol_1d_current > 1.3 * vol_ma_20_aligned[i] and
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, volume surge, ADX > 20
            elif (price_close < s1_aligned[i] and 
                  vol_1d_current > 1.3 * vol_ma_20_aligned[i] and
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to pivot point or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= pivot point or volume < average
                if (price_close <= pp_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= pivot point or volume < average
                if (price_close >= pp_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume1.3x_ADX20"
timeframe = "4h"
leverage = 1.0