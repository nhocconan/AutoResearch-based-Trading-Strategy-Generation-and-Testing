#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R + 1-day Bollinger Band mean reversion with volume confirmation.
In ranging markets (BB width < 50th percentile), fade extreme Williams %R readings (>80 for short, <20 for long).
In trending markets (BB width >= 50th percentile), follow 1-day EMA50 direction.
Volume spike required for entry to avoid low-liquidity false signals.
Designed for low trade frequency by requiring multiple confirmations and regime filter.
Works in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6H data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1D data for regime and trend filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 1D close
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # 50-period EMA on 1D close for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D indicators to 6H timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_std_aligned = align_htf_to_ltf(prices, df_1d, bb_std)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(williams_r[i]) or
            np.isnan(bb_middle_aligned[i]) or np.isnan(bb_std_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: BB width percentile (using 60-day lookback, min 20)
        start_idx = max(0, i - 60)
        bb_width_slice = bb_width_aligned[start_idx:i+1]
        if len(bb_width_slice) >= 20:
            bb_width_percentile = (bb_width_aligned[i] >= bb_width_slice).sum() / len(bb_width_slice) * 100
        else:
            bb_width_percentile = 50  # default to ranging if insufficient data
        
        is_ranging = bb_width_percentile < 50  # BB width below median = ranging
        is_trending = bb_width_percentile >= 50  # BB width above median = trending
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 0:
            if is_ranging and vol_spike:
                # Mean reversion in ranging market: fade extreme Williams %R
                if williams_r[i] < -80:  # Oversold -> long
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20:  # Overbought -> short
                    signals[i] = -0.25
                    position = -1
            elif is_trending and vol_spike:
                # Trend following in trending market: follow 1D EMA50
                if close[i] > ema50_1d_aligned[i]:  # Price above EMA -> long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema50_1d_aligned[i]:  # Price below EMA -> short
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to neutral or trend reverses
                if williams_r[i] > -50 or (is_trending and close[i] < ema50_1d_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to neutral or trend reverses
                if williams_r[i] < -50 or (is_trending and close[i] > ema50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_BBRegime_Volume"
timeframe = "6h"
leverage = 1.0