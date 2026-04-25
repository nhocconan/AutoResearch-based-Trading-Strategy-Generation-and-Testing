#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Regime
Hypothesis: On 6h timeframe, Ichimoku Kumo Twist (Senkou Span A/B cross) with 1d trend filter (price > EMA50 for long, < EMA50 for short) and ADX regime filter (ADX > 25) captures strong trend continuations while avoiding sideways markets. Kumo Twist indicates momentum shift, 1d EMA50 ensures alignment with daily trend, and ADX > 25 filters out low-momentum environments. Designed for 12-37 trades/year to minimize fee drag. Works in bull markets via long entries and bear markets via short entries. Uses discrete position sizing (0.25) to reduce churn. Primary timeframe: 6h, HTF: 1d.
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
    
    # Get 1d data for HTF trend and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_a = np.concatenate([np.full(26, np.nan), senkou_a[:-26]])
    senkou_b = np.concatenate([np.full(26, np.nan), senkou_b[:-26]])
    
    # Calculate ADX on 1d for regime filter
    # +DI, -DI, DX calculation
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 50)  # Ichimoku 52, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        adx_val = adx_aligned[i]
        ema_50_val = ema_50_aligned[i]
        close_val = close[i]
        
        # Kumo Twist: Senkou Span A crosses Senkou Span B
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        if i > start_idx:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
            bullish_twist = (tenkan_val > kijun_val) and (senkou_a_val > senkou_b_val) and (prev_senkou_a <= prev_senkou_b)
            bearish_twist = (tenkan_val < kijun_val) and (senkou_a_val < senkou_b_val) and (prev_senkou_a >= prev_senkou_b)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Trend filter: price relative to EMA50
        uptrend = close_val > ema_50_val
        downtrend = close_val < ema_50_val
        
        # Regime filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Look for entry signals: Kumo Twist with trend and regime filter
            # Long: bullish twist + uptrend + strong trend
            long_signal = bullish_twist and uptrend and strong_trend
            # Short: bearish twist + downtrend + strong trend
            short_signal = bearish_twist and downtrend and strong_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Kumo Twist reversal: bearish twist
            # 2. Trend reversal: price crosses below EMA50
            # 3. Weak trend: ADX drops below 20
            if bearish_twist or (close_val < ema_50_val) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Kumo Twist reversal: bullish twist
            # 2. Trend reversal: price crosses above EMA50
            # 3. Weak trend: ADX drops below 20
            if bullish_twist or (close_val > ema_50_val) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Regime"
timeframe = "6h"
leverage = 1.0