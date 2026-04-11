#!/usr/bin/env python3
"""
6h_12h_1d_cci_extreme_v2
Strategy: 6h CCI extreme reversal with 12h/1d trend filter - refined version
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses CCI(20) extremes (>100 for short, <-100 for long) on 6h combined with 12h EMA50 trend filter and 1d ADX>25 for trend strength. Added minimum holding period (3 bars) and stricter exit conditions to reduce trade frequency. Designed to capture mean reversals in strong trends while avoiding chop. Works in both bull/bear markets by following the higher timeframe trend. Target: 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_extreme_v2"
timeframe = "6h"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h CCI(20)
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma_tp) / (0.015 * md)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        # CCI extreme conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Long: CCI oversold in uptrend with strong trend
        long_signal = cci_oversold and uptrend_12h and strong_trend
        
        # Short: CCI overbought in downtrend with strong trend
        short_signal = cci_overbought and downtrend_12h and strong_trend
        
        # Exit when CCI returns to neutral zone OR minimum holding period exceeded
        exit_long = position == 1 and (cci[i] > -50 or bars_since_entry >= 3)
        exit_short = position == -1 and (cci[i] < 50 or bars_since_entry >= 3)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            bars_since_entry = 0
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            bars_since_entry = 0
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            if position != 0:
                bars_since_entry += 1
    
    return signals

# Hypothesis: Uses CCI(20) extremes (>100 for short, <-100 for long) on 6h combined with 12h EMA50 trend filter and 1d ADX>25 for trend strength. Added minimum holding period (3 bars) and stricter exit conditions to reduce trade frequency. Designed to capture mean reversals in strong trends while avoiding chop. Works in both bull/bear markets by following the higher timeframe trend. Target: 80-120 total trades over 4 years.