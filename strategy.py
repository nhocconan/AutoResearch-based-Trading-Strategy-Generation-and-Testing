#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_RegimeFilter_v1
Hypothesis: Trade daily Donchian(20) breakouts with weekly EMA50 trend filter and choppiness regime filter. Works in bull/bear via weekly trend; chop filter avoids whipsaws in ranging markets. Discrete size 0.25 limits fee drag. Target 15-25 trades/year.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily choppiness index regime filter
    chop_period = 14
    high_roll = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    low_roll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_min_range = high_roll - low_roll
    chop = 100 * np.log10(atr_sum / np.maximum(max_min_range, 1e-10)) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20), ATR(14), chop(14)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Chop regime: avoid extreme chop (range) and extreme trending
        chop_regime_ok = (chop[i] >= 38.2) and (chop[i] <= 61.8)
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + chop regime OK
            long_breakout = close[i] > donchian_high[i]
            long_signal = long_breakout and trend_1w_uptrend and chop_regime_ok
            
            # Short: price breaks below Donchian low + weekly downtrend + chop regime OK
            short_breakout = close[i] < donchian_low[i]
            short_signal = short_breakout and trend_1w_downtrend and chop_regime_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR weekly trend turns down OR ATR-based stoploss
            if (close[i] < donchian_low[i] or not trend_1w_uptrend or
                close[i] < prices['close'].iloc[i-1] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR weekly trend turns up OR ATR-based stoploss
            if (close[i] > donchian_high[i] or not trend_1w_downtrend or
                close[i] > prices['close'].iloc[i-1] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_WeeklyTrend_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0