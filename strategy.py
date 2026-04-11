#!/usr/bin/env python3
"""
4h_1d_cci_extreme_v1
Strategy: 4h CCI extreme reversal with 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses CCI(20) extremes (>100 for short, <-100 for long) on 4h combined with 1d EMA50 trend filter and 1d ADX>25 for trend strength. Designed to capture mean reversals in strong trends while avoiding chop. Works in both bull/bear markets by following the higher timeframe trend. Target: 15-40 trades/year (60-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_extreme_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h CCI(20)
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma_tp) / (0.015 * md)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        # CCI extreme conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Long: CCI oversold in uptrend with strong trend
        long_signal = cci_oversold and uptrend_1d and strong_trend
        
        # Short: CCI overbought in downtrend with strong trend
        short_signal = cci_overbought and downtrend_1d and strong_trend
        
        # Exit when CCI returns to neutral zone
        exit_long = position == 1 and cci[i] > -50
        exit_short = position == -1 and cci[i] < 50
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses CCI(20) extremes (>100 for short, <-100 for long) on 4h combined with 1d EMA50 trend filter and 1d ADX>25 for trend strength. Designed to capture mean reversals in strong trends while avoiding chop. Works in both bull/bear markets by following the higher timeframe trend. Target: 15-40 trades/year (60-160 total over 4 years).