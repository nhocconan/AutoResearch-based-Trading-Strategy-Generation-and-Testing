#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based position sizing.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Entry: Long when price breaks above Donchian(20) high AND 1d EMA34 bullish.
         Short when price breaks below Donchian(20) low AND 1d EMA34 bearish.
- Exit: Opposite Donchian breakout or ATR trailing stop (2.5 * ATR).
- Signal size: 0.30 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels work in both bull and bear markets by capturing breakouts from consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian(20) channels on 4h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donch_high = period20_high[i]
        donch_low = period20_low[i]
        ema_trend = ema_34_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Check for entry signals
            if curr_close > donch_high and curr_close > ema_trend:
                # Long entry: price breaks above Donchian high AND above 1d EMA34
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif curr_close < donch_low and curr_close < ema_trend:
                # Short entry: price breaks below Donchian low AND below 1d EMA34
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Check for exit signals
            # Exit 1: price breaks below Donchian low
            # Exit 2: ATR trailing stop (2.5 * ATR below highest)
            if curr_close < donch_low or curr_close < (highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Check for exit signals
            # Exit 1: price breaks above Donchian high
            # Exit 2: ATR trailing stop (2.5 * ATR above lowest)
            if curr_close > donch_high or curr_close > (lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_ATRStop_v1"
timeframe = "4h"
leverage = 1.0