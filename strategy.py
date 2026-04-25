#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts aligned with 4h EMA34 trend and 1d choppy regime filter.
Only trade when 4h trend is bullish (price > EMA34) for longs or bearish (price < EMA34) for shorts,
AND 1d market is choppy (Choppiness Index > 61.8) to fade false breakouts in ranging markets.
Use session filter (08-20 UTC) to avoid low-volume periods. Position size: 0.20.
Target: 15-35 trades/year to stay within 1h limits and minimize fee drag.
Works in bull (breakouts with trend) and bear (fading false breakouts in chop) markets.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for HTF regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index for regime filter (choppy when CHOP > 61.8)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_period = 14
    chop_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / hl_range) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to neutral if not enough data
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1h Camarilla levels from previous 1h bar
    # We need to compute these from 1h data directly
    # Since we don't have a helper for 1h->1h, we'll compute rolling
    lookback = 2  # need previous bar
    if n < lookback:
        return np.zeros(n)
    
    h_1h = np.roll(high, 1)
    l_1h = np.roll(low, 1)
    c_1h = np.roll(close, 1)
    h_1h[0] = high[0]  # fill first value
    l_1h[0] = low[0]
    c_1h[0] = close[0]
    
    typical_price_1h = (h_1h + l_1h + c_1h) / 3.0
    range_1h = h_1h - l_1h
    camarilla_r1_1h = c_1h + (range_1h * 1.1 / 12.0)   # R1 level
    camarilla_s1_1h = c_1h - (range_1h * 1.1 / 12.0)   # S1 level
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # already DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), and we use previous bar for Camarilla (so +1)
    start_idx = max(34, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA34)
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8) to fade false breakouts
        is_choppy = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 4h uptrend + choppy regime
            long_setup = (close[i] > camarilla_r1_1h[i]) and htf_4h_bullish and is_choppy
            
            # Short setup: price breaks below Camarilla S1 + 4h downtrend + choppy regime
            short_setup = (close[i] < camarilla_s1_1h[i]) and htf_4h_bearish and is_choppy
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches Camarilla S1 (stop) OR 4h trend turns bearish OR regime turns trending
            if (close[i] <= camarilla_s1_1h[i]) or (not htf_4h_bullish) or (not is_choppy):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches Camarilla R1 (stop) OR 4h trend turns bullish OR regime turns trending
            if (close[i] >= camarilla_r1_1h[i]) or (htf_4h_bullish) or (not is_choppy):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0