#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v1
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1d EMA34 trend filter and chop regime filter.
Targets 19-50 trades/year by requiring: 1) price breaks daily R1/S1 levels (strong breakout),
2) aligned with 1d EMA34 trend, 3) choppiness index < 61.8 (trending market). Uses 4h timeframe
to balance trade frequency and fee drag while capturing significant moves in both bull and bear markets.
Chop filter prevents whipsaws in ranging markets, improving performance in bear/range regimes like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/12), S1 = C - 1.1*(HL/12))
    R1 = prev_close + 1.1 * prev_range * (1.0/12.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/12.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Choppiness Index filter (14-period) - loaded ONCE
    chop_period = 14
    true_range = np.maximum(high - low, 
                           np.absolute(high - np.concatenate([[close[0]], close[:-1]])),
                           np.absolute(low - np.concatenate([[close[0]], close[:-1]])))
    atr_sum = pd.Series(true_range).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    chop_filter = chop < 61.8  # Trending market regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for chop calculation (14) and previous day data (1)
    start_idx = max(34, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with chop filter
            # Long breakout: price breaks above R1 with uptrend and trending market
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and chop_filter[i]
            # Short breakout: price breaks below S1 with downtrend and trending market
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and chop_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below S1 (mean reversion) or trend changes or chop becomes high
            if curr_close < S1_aligned[i] or not uptrend or chop_filter[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes or chop becomes high
            if curr_close > R1_aligned[i] or not downtrend or chop_filter[i] == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0