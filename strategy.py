#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_Filter_v1
Hypothesis: Donchian(20) breakout on 6h with 12h EMA20 trend filter targets 12-37 trades/year.
Long when price breaks above 20-period high with 12h uptrend; short when breaks below 20-period low with 12h downtrend.
Uses discrete position sizing (0.25) to minimize fee drag. Works in bull (trend continuation) and bear (mean reversion at extremes) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h data for EMA20 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian channels on 6h (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and 12h EMA20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 12h EMA20
        uptrend = curr_close > ema_20_12h_aligned[i]
        downtrend = curr_close < ema_20_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend alignment
            long_breakout = (curr_close > donchian_high[i]) and uptrend
            short_breakout = (curr_close < donchian_low[i]) and downtrend
            
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
            # Long position: exit on trend reversal or Donchian mean reversion
            if curr_close < ema_20_12h_aligned[i] or curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on trend reversal or Donchian mean reversion
            if curr_close > ema_20_12h_aligned[i] or curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0