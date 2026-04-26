#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 level AND 1d trend is up (close > EMA34) AND market is not in extreme chop (Choppiness Index < 61.8). Enter short when price breaks below Camarilla S1 level AND 1d trend is down (close < EMA34) AND market is not in extreme chop. Uses discrete sizing (0.0, ±0.25) to limit fee churn. Camarilla levels from 1d provide strong support/resistance, 1d trend filter ensures alignment with higher timeframe momentum, and chop filter avoids whipsaws in ranging markets. Designed to generate ~20-30 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes.
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least previous day
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Choppiness Index on 1d (regime filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is NaN
    
    # ATR(14)
    atr_period = 14
    atr = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])  # skip first NaN
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Choppiness Index = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    chop_window = 14
    chop = np.full_like(close_1d_arr, np.nan, dtype=float)
    for i in range(chop_window, len(close_1d_arr)):
        if not np.isnan(atr[i-chop_window+1:i+1]).any():
            sum_atr = np.nansum(atr[i-chop_window+1:i+1])
            if sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr) / np.log10(chop_window)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values  # raw 1d close for Camarilla calculation
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous completed 1d bar to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    
    # First bar has no previous day, set to NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and chop warmup
    start_idx = max(34, 14)  # EMA34 needs 34, chop needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: avoid extreme chop (Choppiness Index > 61.8 = ranging)
        not_extreme_chop = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: breakout above R1 + 1d uptrend + not extreme chop
            long_signal = breakout_up and trend_uptrend and not_extreme_chop
            
            # Short: breakout below S1 + 1d downtrend + not extreme chop
            short_signal = breakout_down and trend_downtrend and not_extreme_chop
            
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
            # Exit: price falls below R1 OR trend change to downtrend OR extreme chop
            if close[i] < r1_aligned[i] or not trend_uptrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above S1 OR trend change to uptrend OR extreme chop
            if close[i] > s1_aligned[i] or not trend_downtrend or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0