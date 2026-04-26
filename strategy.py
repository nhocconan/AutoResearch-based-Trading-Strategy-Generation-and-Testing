#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_1wTrend_VolumeSpike_RegimeFilter
Hypothesis: On daily timeframe, enter long when price breaks above weekly Camarilla R1 AND weekly trend is up (close > weekly EMA34) AND volume > 2.5x 20-day average volume AND market is not excessively choppy (Choppiness Index < 61.8 on daily). Enter short when price breaks below weekly Camarilla S1 AND weekly trend is down (close < weekly EMA34) AND volume spike AND chop filter passes. Exit on weekly trend reversal. Targets BTC/ETH with controlled trade frequency (~15-25/year) by using weekly HTF for structure and daily for execution, minimizing fee drag while capturing sustained trends in both bull and bear markets.
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
    
    # Get weekly data for Camarilla levels and EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    camarilla_range = prev_high_1w - prev_low_1w
    r1 = prev_close_1w + 1.1 * camarilla_range / 12
    s1 = prev_close_1w - 1.1 * camarilla_range / 12
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.5x 20-day average (tighter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.5
    
    # Calculate daily Choppiness Index (14-period) for regime filter
    def true_range(high, low, close_prev):
        return np.maximum(np.maximum(high - low, np.abs(high - close_prev)), np.abs(low - close_prev))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # avoid NaN for first bar
    tr = true_range(high, low, close_prev)
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr14[i] > 0 and not np.isnan(atr14[i]) and not np.isnan(max_high14[i]) and not np.isnan(min_low14[i]):
            chop[i] = 100 * np.log10(np.sum(tr[max(0, i-13):i+1]) / (atr14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral value when not enough data
    
    chop_filter = chop < 61.8  # not excessively choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup, volume MA warmup, and chop calculation
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i])):
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
        
        # Weekly trend filter
        trend_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + weekly uptrend + chop filter
            long_signal = breakout_up and volume_spike[i] and trend_uptrend and chop_filter[i]
            
            # Short: breakout below S1 + volume spike + weekly downtrend + chop filter
            short_signal = breakout_down and volume_spike[i] and trend_downtrend and chop_filter[i]
            
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
            # Exit: weekly trend change to downtrend
            if not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend change to uptrend
            if not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_1wTrend_VolumeSpike_RegimeFilter"
timeframe = "1d"
leverage = 1.0