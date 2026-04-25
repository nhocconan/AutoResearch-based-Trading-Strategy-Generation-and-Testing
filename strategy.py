#!/usr/bin/env python3
"""
6h_ElderRay_Regime_1dTrendFilter_v1
Hypothesis: Use 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and chop regime filter.
Elder Ray measures bull/bear power relative to EMA13. In strong trends (ADX>25) with low chop (CHOP<38.2),
we take trades in direction of 1d trend: long when Bull Power>0, short when Bear Power<0.
Chop filter avoids whipsaws in ranging markets. Discrete sizing 0.25 limits fee drag.
Target: 12-37 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chop regime filter (CHOP(14))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d_arr[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d_arr[:-1]), tr1)
    tr_1d = np.concatenate([[np.inf], tr2])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom_safe)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 6h Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 6h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.inf], tr])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_6h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_6h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(34, 13, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters: only trade in strong trends with low chop
        if adx[i] <= 25 or chop_1d_aligned[i] >= 38.2:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Determine 1d trend from EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(close_1d_aligned):
            signals[i] = 0.0
            continue
            
        if close_1d_aligned > ema_34_1d_aligned[i]:
            daily_trend = 'bullish'  # favor longs
        elif close_1d_aligned < ema_34_1d_aligned[i]:
            daily_trend = 'bearish'  # favor shorts
        else:
            daily_trend = 'neutral'  # no clear trend
        
        if position == 0:
            # Long setup: Bull Power > 0 AND daily trend bullish
            long_setup = (bull_power[i] > 0) and (daily_trend == 'bullish')
            
            # Short setup: Bear Power < 0 AND daily trend bearish
            short_setup = (bear_power[i] < 0) and (daily_trend == 'bearish')
            
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
            # Exit: Bull Power turns negative OR daily trend turns bearish
            if (bull_power[i] <= 0) or (daily_trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR daily trend turns bullish
            if (bear_power[i] >= 0) or (daily_trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Regime_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0