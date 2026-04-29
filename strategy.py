#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and ATR-based position sizing
# Uses Donchian channel from 4h for breakout signals, 1w EMA50 for multi-timeframe trend filter
# ATR(14) for volatility-adjusted stoploss and position sizing (0.25 max)
# Designed for 4h timeframe to balance trade frequency (~25-40/year) and capture medium-term trends
# Works in both bull and bear markets by only taking trades in direction of 1w trend

name = "4h_Donchian20_1wEMA50_ATR_Position_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channel (20-period) from 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss and volatility normalization
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 20, 14)  # EMA50, Donchian, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_atr = atr[i]
        
        # Handle position management
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Stoploss: price closes below highest - 3.0 * ATR
            if curr_close < highest_since_entry - 3.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian low or trend turns down
            elif curr_close < curr_lowest_20 or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: price closes above lowest + 3.0 * ATR
            if curr_close > lowest_since_entry + 3.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian high or trend turns up
            elif curr_close > curr_highest_20 or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high in uptrend (close > EMA50_1w)
            if curr_close > curr_ema50_1w:
                if curr_high > curr_highest_20:  # Break above Donchian high
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                    highest_since_entry = curr_high
            # Short entry: price breaks below Donchian low in downtrend (close < EMA50_1w)
            elif curr_close < curr_ema50_1w:
                if curr_low < curr_lowest_20:  # Break below Donchian low
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                    lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals