#!/usr/bin/env python3
"""
1d_Choppiness_Index_Regime_MeanReversion
Hypothesis: Use Choppiness Index (14) to detect ranging markets (CHOP > 61.8) and mean-revert at Bollinger Bands (20,2). In trending markets (CHOP < 38.2), follow the trend using EMA(50). Works in both bull and bear markets by adapting to regime. Designed for 1d timeframe to limit trades (<25/year) and avoid fee drag.
"""

name = "1d_Choppiness_Index_Regime_MeanReversion"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Calculate +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # Calculate DX and ADX(14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align indicators to lower timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Ranging market: CHOP > 61.8 -> mean revert at Bollinger Bands
            if chop_val > 61.8:
                if close[i] <= lower_bb_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # Trending market: CHOP < 38.2 and ADX > 25 -> follow EMA trend
            elif chop_val < 38.2 and adx_val > 25:
                if close[i] > sma_20_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < sma_20_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CHOP drops below 38.2 (trend start) or price crosses above SMA20 in range
            if chop_aligned[i] < 38.2 or close[i] >= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CHOP drops below 38.2 (trend start) or price crosses below SMA20 in range
            if chop_aligned[i] < 38.2 or close[i] <= sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals