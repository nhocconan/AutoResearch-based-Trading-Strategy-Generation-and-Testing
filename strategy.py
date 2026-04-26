#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and 1d chop regime filter. Uses discrete position sizing (0.20) to limit drawdown. Targets 15-37 trades/year by requiring confluence of: 1) price breaking Camarilla levels from prior 4h bar, 2) 4h EMA50 trend alignment, 3) 1d chop index < 50 (trending regime). Works in bull (breakouts with trend) and bear (fade at extremes with volume/volatility confirmation via regime filter). Designed for low trade frequency to overcome fee drag in ranging/bear markets like 2025+.
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
    
    # Get 4h data for Camarilla calculation and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Avoid NaN from shift - use current bar if previous not available
    prev_high = np.where(np.isnan(prev_high), df_4h['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_4h['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_4h['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate Chopiness Index on 1d for regime filter (trending when < 50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    hl_range_14 = hh_14 - ll_14
    chop_1d = np.full_like(close_1d, 50.0)  # default to neutral
    valid = (hl_range_14 > 0) & (~np.isnan(hl_range_14)) & (~np.isnan(sum_tr_14))
    chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / hl_range_14[valid]) / np.log10(14)
    
    # Align Chopiness Index to 1h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(50), 1d chop calculation (need 28 for 14*2)
    start_idx = max(50, 28) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        chop_regime = chop_1d_aligned[i] < 50.0  # trending regime (chop < 50)
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend up AND trending regime
            long_signal = (close_val > r1_aligned[i]) and trend_4h_up and chop_regime
            
            # Short: price breaks below S1 AND 4h trend down AND trending regime
            short_signal = (close_val < s1_aligned[i]) and trend_4h_down and chop_regime
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend flips down OR chop regime becomes ranging (chop >= 50)
            if (not trend_4h_up) or (not chop_regime):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR chop regime becomes ranging (chop >= 50)
            if (not trend_4h_down) or (not chop_regime):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0