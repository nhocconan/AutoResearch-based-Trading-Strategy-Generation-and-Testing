#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_1d_atr_v2
# Hypothesis: Weekly pivot points act as strong support/resistance on 6h timeframe.
# Breakout above weekly R1 with volume confirmation and ATR filter = long.
# Breakdown below weekly S1 with volume confirmation and ATR filter = short.
# Uses 1d ATR for volatility regime filter (only trade when ATR > 20-period MA).
# Works in bull/bear: pivots adapt to price levels, ATR filter avoids choppy markets.
# Target: 12-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_1d_atr_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 1d HTF data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with same length
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR regime to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when ATR > 20-period MA
        volatility_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly pivot OR volatility dies
            if close[i] < weekly_pivot_aligned[i] or not volatility_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly pivot OR volatility dies
            if close[i] > weekly_pivot_aligned[i] or not volatility_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if not volatility_filter:
                signals[i] = 0.0
                continue
                
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Breakout above weekly R1
                if close[i] > weekly_r1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below weekly S1
                elif close[i] < weekly_s1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals

# Align weekly pivot points (need to compute after getting aligned arrays)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)