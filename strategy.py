#!/usr/bin/env python3
"""
6h_IchiCloud_TKCross_1dTrend_ADXFilter_v1
Hypothesis: On 6h timeframe, trade Ichimoku TK cross signals with 1d EMA50 trend filter and ADX(14) > 25 regime filter.
Ichimoku provides dynamic support/resistance via cloud, TK cross for momentum, daily trend for bias, ADX to avoid chop.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear markets via daily trend filter and ADX regime filter.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # TK Cross signals
    tk_cross_above = (tenkan_sen > kijun_sen) & (tenkan_sen.shift(1) <= kijun_sen.shift(1))
    tk_cross_below = (tenkan_sen < kijun_sen) & (tenkan_sen.shift(1) >= kijun_sen.shift(1))
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d ADX(14) for regime filter
    # ADX calculation: +DI, -DI, DX
    period14_high = df_1d['high'].values
    period14_low = df_1d['low'].values
    period14_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(period14_high[1:] - period14_low[1:])
    tr2 = np.abs(period14_high[1:] - period14_close[:-1])
    tr3 = np.abs(period14_low[1:] - period14_close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # +DM and -DM
    up_move = period14_high[1:] - period14_high[:-1]
    down_move = period14_low[:-1] - period14_low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def WilderSmoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return smoothed
        # First value is simple average
        smoothed[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(values)):
            smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    atr_14 = WilderSmoothing(tr, 14)
    plus_di_14 = 100 * WilderSmoothing(plus_dm, 14) / atr_14
    minus_di_14 = 100 * WilderSmoothing(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = WilderSmoothing(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    tk_cross_above_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_above.astype(float))
    tk_cross_below_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_below.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), EMA50, ADX
    start_idx = 52 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tk_cross_above_aligned[i]) or
            np.isnan(tk_cross_below_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        adx_14_val = adx_14_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        tk_above = tk_cross_above_aligned[i] > 0.5
        tk_below = tk_cross_below_aligned[i] > 0.5
        
        # Cloud: green (bullish) when span_a > span_b, red (bearish) when span_a < span_b
        cloud_bullish = span_a > span_b
        cloud_bearish = span_a < span_b
        
        # Price above/below cloud
        price_above_cloud = close_val > max(span_a, span_b)
        price_below_cloud = close_val < min(span_a, span_b)
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_14_val > 25
        
        if position == 0:
            # Long: TK cross above + price above cloud + uptrend + trending regime
            long_signal = tk_above and price_above_cloud and uptrend and trending
            
            # Short: TK cross below + price below cloud + downtrend + trending regime
            short_signal = tk_below and price_below_cloud and downtrend and trending
            
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
            # Exit: TK cross below OR price drops below cloud OR trend reversal
            if tk_below or close_val < min(span_a, span_b) or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above OR price rises above cloud OR trend reversal
            if tk_above or close_val > max(span_a, span_b) or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_IchiCloud_TKCross_1dTrend_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0