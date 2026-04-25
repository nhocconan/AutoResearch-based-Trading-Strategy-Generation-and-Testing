#!/usr/bin/env python3
"""
1d_Camarilla_H1L1_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla H1/L1 breakouts on daily timeframe with 1-week EMA34 trend filter and choppiness regime filter (CHOP < 40).
H1/L1 are tighter breakout levels than H3/L3 or R3/S3, capturing intraday momentum within the daily bar.
Only trade when 1w trend is aligned (price > EMA34 for long, price < EMA34 for short) AND market is trending (Choppiness Index < 40).
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.25 to balance profit and fee drag.
Target: 15-25 trades/year to stay well under 150-trade 1d hard max.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla levels (same timeframe as primary)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_h1_1d = c_1d + (range_1d * 1.1 / 12.0)   # H1 level
    camarilla_l1_1d = c_1d - (range_1d * 1.1 / 12.0)   # L1 level
    
    # Align Camarilla levels to 1d timeframe (use previous 1d bar's levels)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1_1d)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1_1d)
    
    # Calculate 1d Choppiness Index for regime filter (trending when CHOP < 40)
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
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA34)
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 40)
        is_trending = chop[i] < 40.0
        
        if position == 0:
            # Long setup: price breaks above Camarilla H1 + 1w uptrend + trending regime
            long_setup = (close[i] > camarilla_h1_aligned[i]) and htf_1w_bullish and is_trending
            
            # Short setup: price breaks below Camarilla L1 + 1w downtrend + trending regime
            short_setup = (close[i] < camarilla_l1_aligned[i]) and htf_1w_bearish and is_trending
            
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
            # Exit: price touches Camarilla L1 (stop) OR 1w trend turns bearish OR regime turns choppy
            if (close[i] <= camarilla_l1_aligned[i]) or (not htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H1 (stop) OR 1w trend turns bullish OR regime turns choppy
            if (close[i] >= camarilla_h1_aligned[i]) or (htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H1L1_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0