#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivotFilter
Hypothesis: 6-hour Donchian(20) breakout with 1-day EMA50 trend filter and weekly Camarilla H3/L3 as regime filter.
Targets 12-30 trades/year by requiring: 1) price breaks 6h Donchian(20) levels, 2) aligned with 1d EMA50 trend,
3) price not in weekly Camarilla extreme zone (H3/L3) to avoid exhaustion moves. Uses 6h timeframe to reduce
fee drag while capturing multi-day trends. Weekly pivot filter avoids counter-trend breakouts in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA50 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w data for weekly Camarilla H3/L3 (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # Weekly Camarilla: based on previous week OHLC
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_range = prev_week_high - prev_week_low
    # H3 = C + 1.1*(HL/2), L3 = C - 1.1*(HL/2)
    H3 = prev_week_close + 1.1 * prev_week_range * (1.0/2.0)
    L3 = prev_week_close - 1.1 * prev_week_range * (1.0/2.0)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    
    # 6h Donchian(20) - use rolling window on 6h data directly
    # Need at least 20 periods for Donchian
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA50 (50) + 1w data (1) + Donchian(20)
    start_idx = 50 + 1 + 20  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Regime filter: avoid extreme weekly Camarilla zones (exhaustion)
        # In uptrend, avoid if price >= H3 (overbought)
        # In downtrend, avoid if price <= L3 (oversold)
        not_overbought = curr_close < H3_aligned[i]
        not_oversold = curr_close > L3_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend alignment and regime filter
            # Long breakout: price breaks above Donchian high with uptrend and not overbought
            long_breakout = (curr_close > donchian_high[i]) and uptrend and not_overbought
            # Short breakout: price breaks below Donchian low with downtrend and not oversold
            short_breakout = (curr_close < donchian_low[i]) and downtrend and not_oversold
            
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
            # Exit if price breaks below Donchian low (mean reversion) or trend changes to downtrend
            if curr_close < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above Donchian high (mean reversion) or trend changes to uptrend
            if curr_close > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivotFilter"
timeframe = "6h"
leverage = 1.0