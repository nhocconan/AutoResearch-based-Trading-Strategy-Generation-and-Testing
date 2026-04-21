#!/usr/bin/env python3
"""
Hypothesis: 6h weekly pivot breakout with volume confirmation and daily ATR filter.
Long when price breaks above weekly R1 with volume > 1.5x average and daily ATR > daily ATR mean;
Short when price breaks below weekly S1 with volume > 1.5x average and daily ATR > daily ATR mean.
Exit when price returns to weekly pivot or 1.5x ATR stop. Weekly pivot levels represent key
institutional support/resistance. Volume surge confirms breakout validity. ATR filter ensures
sufficient volatility for meaningful moves. Designed for 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Standard pivot point calculation
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # R1 and S1 are the first resistance/support levels
    r1 = pivot + range_1w * 0.1
    s1 = pivot - range_1w * 0.1
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Load daily data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR (14-period)
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
    
    # Daily ATR mean (20-period for filter)
    atr_mean_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_mean_20_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_20)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR for stop (14-period on 6h)
    tr1_6h = prices['high'].values - prices['low'].values
    tr2_6h = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3_6h = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(atr_mean_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily values aligned to 6h
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        atr_1d_current = atr_1d_aligned[i]
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above weekly R1 with volume surge and sufficient volatility
            if (price_high > r1_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                atr_1d_current > atr_mean_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly S1 with volume surge and sufficient volatility
            elif (price_low < s1_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  atr_1d_current > atr_mean_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: return to weekly pivot or 1.5x ATR stop
            exit_signal = False
            
            if position == 1:
                # Exit long: touch weekly pivot OR price < entry - 1.5*ATR
                if price_low <= pivot_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use R1 as entry level for long
                    entry_level = r1_aligned[i-1] if i >= 1 else r1_aligned[0]
                    if price_close < entry_level - 1.5 * atr_6h[i]:
                        exit_signal = True
            elif position == -1:
                # Exit short: touch weekly pivot OR price > entry + 1.5*ATR
                if price_high >= pivot_aligned[i]:
                    exit_signal = True
                else:
                    # Track entry approximation: use S1 as entry level for short
                    entry_level = s1_aligned[i-1] if i >= 1 else s1_aligned[0]
                    if price_close > entry_level + 1.5 * atr_6h[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_DailyATR_Volume1.5x"
timeframe = "6h"
leverage = 1.0