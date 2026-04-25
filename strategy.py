#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime filter. Long when price breaks above R1 in uptrend (price > 1d EMA50) and chop < 61.8 (trending market). Short when price breaks below S1 in downtrend (price < 1d EMA50) and chop < 61.8. Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-40 trades/year per symbol, effective in trending markets while avoiding range-bound whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * (1.0/12.0)  # R1 = C + 1.1*(H-L)/12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * (1.0/12.0)  # S1 = C - 1.1*(H-L)/12
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate Choppiness Index on 1d
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(window)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, window=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 1d EMA50) and trending market (chop < 61.8)
            # Short: price breaks below Camarilla S1 in downtrend (price < 1d EMA50) and trending market (chop < 61.8)
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_aligned[i]) and (chop_aligned[i] < 61.8)
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_aligned[i]) and (chop_aligned[i] < 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 1d EMA50 (trend reversal) OR chop > 61.8 (range market)
            exit_signal = (close[i] < ema_50_aligned[i]) or (chop_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal) OR chop > 61.8 (range market)
            exit_signal = (close[i] > ema_50_aligned[i]) or (chop_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0