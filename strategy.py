#!/usr/bin/env python3
"""
6H_Choppiness_Regime_Position
Hypothesis: Use Choppiness Index to detect regimes (trending/ranging) and apply appropriate strategies.
- Trending (CHOP < 38.2): Follow trend with EMA crossover (EMA12 > EMA26 = long, < = short)
- Ranging (CHOP > 61.8): Mean revert at Bollinger Bands (price < BB lower = long, > BB upper = short)
- Neutral zone: No position
Weekly trend filter avoids counter-trend trades. Designed for 6h timeframe to balance trade frequency and robustness.
Works in bull markets (trend following) and bear markets (avoids longs in downtrends, takes shorts in downtrends via regime).
Targets 12-37 trades/year to minimize fee drag on 6h timeframe.
"""
name = "6H_Choppiness_Regime_Position"
timeframe = "6h"
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
    
    # Get 6H data for Choppiness Index calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 6H data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0.0]), tr])  # align length
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.where((hh - ll) != 0, 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14), 50)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # adjust for rolling window
    chop_6h_aligned = align_htf_to_ltf(prices, df_6h, chop)
    
    # EMA for trend following (12, 26) on 6H close
    ema12 = pd.Series(close_6h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_6h).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema12_6h_aligned = align_htf_to_ltf(prices, df_6h, ema12)
    ema26_6h_aligned = align_htf_to_ltf(prices, df_6h, ema26)
    
    # Bollinger Bands (20, 2) on 6H close
    sma20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_upper_6h_aligned = align_htf_to_ltf(prices, df_6h, bb_upper)
    bb_lower_6h_aligned = align_htf_to_ltf(prices, df_6h, bb_lower)
    
    # Weekly trend filter: EMA50 on 1W close
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(chop_6h_aligned[i]) or np.isnan(ema12_6h_aligned[i]) or 
            np.isnan(ema26_6h_aligned[i]) or np.isnan(bb_upper_6h_aligned[i]) or 
            np.isnan(bb_lower_6h_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_6h_aligned[i]
        ema12_val = ema12_6h_aligned[i]
        ema26_val = ema26_6h_aligned[i]
        bb_upper_val = bb_upper_6h_aligned[i]
        bb_lower_val = bb_lower_6h_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        if position == 0:
            # Trending regime: follow EMA crossover
            if chop_val < 38.2:
                if ema12_val > ema26_val and close[i] > ema50_1w_val:
                    signals[i] = 0.25
                    position = 1
                elif ema12_val < ema26_val and close[i] < ema50_1w_val:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: mean revert at Bollinger Bands
            elif chop_val > 61.8:
                if close[i] < bb_lower_val:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > bb_upper_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend regime broken or reverse signal
            if chop_val < 38.2 and ema12_val < ema26_val:
                signals[i] = 0.0
                position = 0
            elif chop_val > 61.8 and close[i] > sma20:  # return to mean in ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend regime broken or reverse signal
            if chop_val < 38.2 and ema12_val > ema26_val:
                signals[i] = 0.0
                position = 0
            elif chop_val > 61.8 and close[i] < sma20:  # return to mean in ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals