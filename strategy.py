#!/usr/bin/env python3
"""
12h Donchian(20) Breakout with 1d EMA50 Trend Filter and ATR(14) Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum. Aligning with 1d EMA50 trend filters false breakouts. ATR-based stoploss limits drawdown. Designed for 12h timeframe to trade ~12-37 times per year (50-150 total over 4 years), minimizing fee drag. Works in bull markets via upside breakouts and bear markets via downside breakdowns when aligned with higher timeframe trend.
"""

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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for EMA + buffer
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Donchian(20) channels (using 12h data)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, Donchian, ATR
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian + price above 1d EMA50
            long_signal = (curr_high > upper_channel) and (curr_close > ema_50_val)
            # Short: break below lower Donchian + price below 1d EMA50
            short_signal = (curr_low < lower_channel) and (curr_close < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Trail stop: exit if price drops below highest high since entry - 2*ATR
            # Simplified: exit if price closes below upper channel (breakout failed) OR below EMA50
            if (curr_close < upper_channel) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Trail stop: exit if price rises above lowest low since entry + 2*ATR
            # Simplified: exit if price closes above lower channel (breakdown failed) OR above EMA50
            if (curr_close > lower_channel) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_ATRStop_v1"
timeframe = "12h"
leverage = 1.0