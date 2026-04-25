#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade breakouts of 1d Camarilla H3/L3 levels on 12h timeframe with 1d EMA34 trend filter and choppiness regime filter.
In bull markets: buy when price breaks above H3 and price > EMA34 and chop < 61.8.
In bear markets: sell when price breaks below L3 and price < EMA34 and chop < 61.8.
Chop filter avoids whipsaws in ranging markets. Position size: 0.25.
Target: 12-37 trades/year to stay under 200-trade 12h hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Get 1d data for HTF trend filter, Camarilla levels, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 bars for chop calculation
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Previous day close for Camarilla
    
    # Camarilla calculation: based on previous day's range
    rang = high_1d_prev - low_1d_prev
    H3 = close_1d_prev + rang * 1.1 / 4
    L3 = close_1d_prev - rang * 1.1 / 4
    H4 = close_1d_prev + rang * 1.1 / 2
    L4 = close_1d_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    # We'll use a simplified version: high-low range vs true range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl[range_hl == 0] = 1e-10
    
    chop = 100 * np.log10(atr * 14 / range_hl) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and CHOP (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: only trade when not too choppy (trending market)
        regime_ok = chop_aligned[i] < 61.8  # Below chop threshold = trending
        
        if position == 0:
            # Long setup: price breaks above H3 + 1d uptrend + trending regime
            long_setup = (close[i] > H3_aligned[i]) and htf_1d_bullish and regime_ok
            
            # Short setup: price breaks below L3 + 1d downtrend + trending regime
            short_setup = (close[i] < L3_aligned[i]) and htf_1d_bearish and regime_ok
            
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
            # Exit: price touches L3 (stop) OR 1d trend turns bearish OR chop too high
            if (close[i] <= L3_aligned[i]) or (not htf_1d_bullish) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H3 (stop) OR 1d trend turns bullish OR chop too high
            if (close[i] >= H3_aligned[i]) or (htf_1d_bullish) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0