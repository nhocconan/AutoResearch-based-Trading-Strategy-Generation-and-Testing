#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 6h with 1d EMA34 trend filter and choppiness regime filter (CHOP < 40).
R3/S3 are stronger breakout levels than H3/L3 or R1/S1, reducing false breakouts in choppy markets.
Only trade when 1d trend is aligned (price > EMA34 for long, price < EMA34 for short) AND market is trending (Choppiness Index < 40).
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.25 to balance profit and fee drag.
Target: 15-25 trades/year to stay well under 300-trade 6h hard max.
Works in bull (breakouts with trend) and bear (strong breakdowns with trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_r3_1d = c_1d + (range_1d * 1.1 / 2.0)   # R3 level
    camarilla_s3_1d = c_1d - (range_1d * 1.1 / 2.0)   # S3 level
    
    # Align Camarilla levels to 6h timeframe (use previous 1d bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 6h Choppiness Index for regime filter (trending when CHOP < 40)
    atr_period = 14
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / hl_range) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to neutral if not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), ATR (14), and CHOP (14)
    start_idx = max(34, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 40)
        is_trending = chop[i] < 40.0
        
        if position == 0:
            # Long setup: price breaks above Camarilla R3 + 1d uptrend + trending regime
            long_setup = (close[i] > camarilla_r3_aligned[i]) and htf_1d_bullish and is_trending
            
            # Short setup: price breaks below Camarilla S3 + 1d downtrend + trending regime
            short_setup = (close[i] < camarilla_s3_aligned[i]) and htf_1d_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S3 (stop) OR 1d trend turns bearish OR regime turns choppy
            if (close[i] <= camarilla_s3_aligned[i]) or (not htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 (stop) OR 1d trend turns bullish OR regime turns choppy
            if (close[i] >= camarilla_r3_aligned[i]) or (htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0