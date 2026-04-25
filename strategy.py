#!/usr/bin/env python3
"""
6h_ElderRay_1dRegime_Filter
Hypothesis: Trade Elder Ray bull/bear power on 6h with 1d regime filter (ADX + EMA200) to avoid whipsaws.
Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
Long when Bull Power > 0 and rising + regime bullish.
Short when Bear Power > 0 and rising + regime bearish.
Regime: ADX > 25 + EMA200 slope for trend, else range (fade extremes).
Target: 12-30 trades/year (50-120 over 4 years) to minimize fee drag.
Works in bull (trend follow) and bear (range fade) via regime adaptation.
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
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = np.diff(df_1d['high'].values)
    minus_dm = np.diff(df_1d['low'].values)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = np.maximum(np.abs(np.diff(df_1d['high'].values)),
                    np.maximum(np.abs(np.diff(df_1d['low'].values)),
                               np.abs(np.diff(df_1d['close'].values))))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA200 for trend direction
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_slope = np.gradient(ema_200_1d)  # slope of EMA200
    
    # Align regime indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_200_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_200_slope)
    
    # Calculate Elder Ray on 6h: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Smooth bull/bear power with EMA3 to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Momentum of smoothed power (rising if current > previous)
    bull_power_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_power_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and 1d indicators
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(ema_200_slope_aligned[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        is_trending = adx_1d_aligned[i] > 25
        is_bull_trend = is_trending and (close[i] > ema_200_1d_aligned[i]) and (ema_200_slope_aligned[i] > 0)
        is_bear_trend = is_trending and (close[i] < ema_200_1d_aligned[i]) and (ema_200_slope_aligned[i] < 0)
        is_range = not is_trending  # ADX <= 25 = range
        
        if position == 0:
            # Long setup: Bull Power > 0 and rising + (trend bullish OR range with bullish bias)
            long_setup = (bull_power_smooth[i] > 0) and bull_power_rising[i] and \
                         (is_bull_trend or (is_range and close[i] > ema_200_1d_aligned[i]))
            # Short setup: Bear Power > 0 and rising + (trend bearish OR range with bearish bias)
            short_setup = (bear_power_smooth[i] > 0) and bear_power_rising[i] and \
                          (is_bear_trend or (is_range and close[i] < ema_200_1d_aligned[i]))
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR regime turns bearish
            if (bull_power_smooth[i] <= 0) or is_bear_trend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power <= 0 OR regime turns bullish
            if (bear_power_smooth[i] <= 0) or is_bull_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_1dRegime_Filter"
timeframe = "6h"
leverage = 1.0