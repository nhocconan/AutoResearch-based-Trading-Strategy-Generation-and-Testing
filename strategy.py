#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Regime_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (price > 1d EMA34 for long, < 1d EMA34 for short) and choppiness regime filter (CHOP > 61.8 = range -> mean reversion at S1/R1, CHOP < 38.2 = trend -> breakout at R3/S3). Uses 1d for HTF alignment. Designed to work in bull markets (breakout mode) and bear markets (mean reversion in range) via regime adaptation. Targets 12-30 trades/year by requiring regime confirmation and avoiding whipsaw.
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
    
    # Get 1d data for HTF trend and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Choppiness Index on 1d
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    
    # Align HTF indicators to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    camarilla_r1 = close_1d + 1.1/12 * (high_1d - low_1d)  # R1 = C + 1.1/12*(H-L)
    camarilla_s1 = close_1d - 1.1/12 * (high_1d - low_1d)  # S1 = C - 1.1/12*(H-L)
    
    # Align Camarilla levels to 12h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        chop_val = chop_aligned[i]
        is_trending = chop_val < 38.2  # Trending regime
        is_ranging = chop_val > 61.8   # Ranging regime
        
        if position == 0:
            if is_trending:
                # Trending regime: breakout at R3/S3 with 1d EMA34 filter
                long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema34_aligned[i])
                short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema34_aligned[i])
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean reversion at S1/R1
                long_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] > camarilla_s3_aligned[i])
                short_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] < camarilla_r3_aligned[i])
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trades
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            exit_signal = False
            if is_trending:
                # Exit on trend reversal below EMA34
                if close[i] < ema34_aligned[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit at R1 (mean reversion target)
                if close[i] > camarilla_r1_aligned[i]:
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            exit_signal = False
            if is_trending:
                # Exit on trend reversal above EMA34
                if close[i] > ema34_aligned[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit at S1 (mean reversion target)
                if close[i] < camarilla_s1_aligned[i]:
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0