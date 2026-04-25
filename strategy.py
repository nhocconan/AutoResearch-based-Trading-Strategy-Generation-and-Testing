#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: 12-hour Donchian(20) breakout with 1-week EMA50 trend filter and choppiness regime filter.
Targets 12-37 trades/year by requiring: 1) price breaks 20-period Donchian channel on 12h,
2) aligned with weekly EMA50 trend, 3) choppiness index > 50 (range/transition regime avoids strong trends where breakouts fail).
Uses discrete position sizing (0.25) to minimize fee drag. Works in bull/bear via trend filter and regime avoidance of whipsaw markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w data for choppiness index (loaded ONCE)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) for 1w
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over last 14 periods
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: 100 * log10(sum_atr_14 / range_14) / log10(14)
    # Avoid division by zero and log of zero
    chop_raw = np.divide(sum_atr_14, range_14, out=np.full_like(sum_atr_14, np.nan), where=range_14!=0)
    chop_raw = np.where((chop_raw > 0) & (range_14 > 0), chop_raw, np.nan)
    chop_1w = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    # 12h Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1w EMA50 (50) and choppiness calculation (14+14-1=27) -> max 50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Regime filter: choppiness > 50 (avoid strong trending markets where breakouts fail)
        regime_ok = chop_1w_aligned[i] > 50
        
        if position == 0:
            # Look for entry signals with trend alignment and regime filter
            # Long breakout: price breaks above Donchian high with uptrend and regime OK
            long_breakout = (curr_close > donchian_high[i]) and uptrend and regime_ok
            # Short breakout: price breaks below Donchian low with downtrend and regime OK
            short_breakout = (curr_close < donchian_low[i]) and downtrend and regime_ok
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below Donchian low or trend changes
            if (curr_close < donchian_low[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above Donchian high or trend changes
            if (curr_close > donchian_high[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0