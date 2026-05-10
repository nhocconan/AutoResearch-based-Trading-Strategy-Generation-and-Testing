#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_VolumeRegime
# Hypothesis: Enter long/short at Camarilla R1/S1 levels when price breaks above/below with volume confirmation, 
# filtered by 1d trend (EMA34) and Choppiness regime (range-bound markets favor mean reversion). 
# Works in bull/bear markets: buys dips in uptrends, sells rallies in downtrends; avoids whipsaws via regime filter.
# Targets ~25 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_VolumeRegime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels, trend, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels (using previous day)
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = prev_close > ema34_1d
    trend_1d_down = prev_close < ema34_1d
    
    # Choppiness Index (14-period) for regime filter
    atr_1d = pd.Series(np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                                               np.abs(df_1d['high'] - df_1d['close'].shift(1))),
                                   np.abs(df_1d['low'] - df_1d['close'].shift(1)))).rolling(14, min_periods=14).mean().values
    sum_true_range = pd.Series(atr_1d).rolling(14, min_periods=14).sum().values
    highest_high = df_1d['high'].rolling(14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(14, min_periods=14).min().values
    chop = 100 * np.log10(sum_true_range / (highest_high - lowest_low)) / np.log10(14)
    chop_mask = chop > 61.8  # Range-bound regime (favor mean reversion)
    
    # Align 1d indicators to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_mask.astype(float))
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike in ranging market
            if (close[i] > R1_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                vol_spike[i] and
                chop_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume spike in ranging market
            elif (close[i] < S1_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  vol_spike[i] and
                  chop_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price re-enters below R1 or trend reverses
            if (close[i] < R1_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price re-enters above S1 or trend reverses
            if (close[i] > S1_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals