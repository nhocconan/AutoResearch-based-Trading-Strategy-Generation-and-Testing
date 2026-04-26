#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_ADX_Filter_v1
Hypothesis: 6h Ichimoku cloud twist (Senkou Span A/B cross) with ADX regime filter and 1d HTF trend alignment. 
Cloud twist indicates momentum shift; ADX>25 confirms strength to follow twist direction. 
In ranging markets (ADX<20), fade extreme price deviations from Kumo edges. 
Uses discrete sizing (0.25) to minimize fees. Targets 50-150 trades over 4 years.
Works in bull/bear via adaptive logic: trend following in strong trends, mean reversion in chop.
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
    
    # Load 1d data ONCE before loop for HTF trend and Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Kumo twist detection: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = senkou_a[0]
    senkou_b_prev[0] = senkou_b[0]
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Kumo (cloud) boundaries for mean reversion: use current Senkou Span A/B
    kumō_top = np.maximum(senkou_a, senkou_b)  # Upper cloud boundary
    kumō_bottom = np.minimum(senkou_a, senkou_b)  # Lower cloud boundary
    
    # ADX calculation for regime filtering
    adx_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr_14 = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 14 for ADX)
    start_idx = max(52, adx_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or
            np.isnan(senkou_b[i]) or np.isnan(adx[i]) or np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime-based logic
        if adx[i] > 25:  # Trending regime
            # Follow Kumo twist direction aligned with HTF trend
            if bullish_twist[i] and htf_trend[i] == 1:  # Bullish twist in uptrend HTF
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif bearish_twist[i] and htf_trend[i] == -1:  # Bearish twist in downtrend HTF
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # No clear twist signal or regime mismatch - exit
                signals[i] = 0.0
                position = 0
        elif adx[i] < 20:  # Ranging regime
            # Mean revert at Kumo edges (cloud boundaries)
            if close[i] < kumō_bottom[i] and htf_trend[i] == 1:  # Long mean reversion from lower cloud in uptrend HTF
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] > kumō_top[i] and htf_trend[i] == -1:  # Short mean reversion from upper cloud in downtrend HTF
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion position when price returns to cloud
                if position == 1 and close[i] > kumō_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] < kumō_top[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:  # Transition regime (20 <= ADX <= 25)
            # Hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0