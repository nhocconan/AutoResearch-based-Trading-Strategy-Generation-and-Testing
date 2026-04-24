#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 12h ATR-based volatility filter + 1d EMA34 trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ATR(14) for volatility regime (only trade when ATR ratio > 1.2, indicating expanded volatility).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend bias, price < EMA34 = downtrend bias).
- Entry: Long when price breaks above Donchian(20) upper band AND ATR ratio > 1.2 AND price > 1d EMA34;
         Short when price breaks below Donchian(20) lower band AND ATR ratio > 1.2 AND price < 1d EMA34.
- Exit: Reverse signal on opposite Donchian breakout OR ATR ratio < 0.8 (volatility contraction).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture momentum; volatility filter ensures trading only during energetic moves;
  EMA34 provides higher-timeframe trend alignment to reduce counter-trend whipsaws.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volatility timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ATR(14) volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period TR is just high-low
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h ATR(50) for ratio (longer-term volatility)
    atr_50_12h = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_12h / np.where(atr_50_12h == 0, 1, atr_50_12h)  # avoid div by zero
    
    # Align ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 50)  # Donchian needs 20, EMA34 needs 34, ATR ratio needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        
        # Volatility filter: only trade when ATR ratio > 1.2 (expanded volatility)
        vol_expanded = curr_atr_ratio > 1.2
        vol_contracting = curr_atr_ratio < 0.8
        
        # Trend filter from 1d EMA34
        uptrend_bias = curr_close > ema_34_aligned[i]
        downtrend_bias = curr_close < ema_34_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > highest_high[i-1]  # break above previous period's high
        breakout_down = curr_low < lowest_low[i-1]   # break below previous period's low
        
        if position == 0:
            # Check for entry signals
            if vol_expanded and uptrend_bias and breakout_up:
                signals[i] = 0.25
                position = 1
            elif vol_expanded and downtrend_bias and breakout_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit on reverse breakout OR volatility contraction
            if breakout_down or vol_contracting:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on reverse breakout OR volatility contraction
            if breakout_up or vol_contracting:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hATR_VolFilter_1dEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0