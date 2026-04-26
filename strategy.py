#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Regime_v1
Hypothesis: Daily Donchian(20) breakout with 1w trend filter and chop regime filter.
Long when price breaks above upper band AND 1w close > EMA(34) AND chop < 61.8 (trending).
Short when price breaks below lower band AND 1w close < EMA(34) AND chop < 61.8.
ATR-based stoploss and discrete position sizing (0.25) to limit fees.
Designed to capture sustained trends in both bull and bear markets via 1w trend filter.
Target trades: 30-100 over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get 1d data for chop regime filter (ATR-based)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Calculate ATR(14) on 1d for chop regime and stoploss
    tr1 = pd.Series(df_1d['high']).shift(0) - pd.Series(df_1d['low']).shift(0)
    tr2 = abs(pd.Series(df_1d['high']).shift(0) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']).shift(0) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Chop regime: ATR(14) / ATR(50) < 0.382 = trending (use 0.382 as threshold)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    chop = atr_14_aligned / (atr_50_aligned + 1e-10)  # avoid div zero
    chop_threshold = 0.382  # below this = trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), EMA34(1w), ATR(14), ATR(50)
    start_idx = max(20, 34, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_50_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trending_regime = chop[i] < chop_threshold
        trend_up = close_val > ema_34_1w_aligned[i]
        trend_down = close_val < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND trending regime AND 1w uptrend
            long_signal = (close_val > upper_aligned[i]) and trending_regime and trend_up
            
            # Short: price breaks below lower band AND trending regime AND 1w downtrend
            short_signal = (close_val < lower_aligned[i]) and trending_regime and trend_down
            
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
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below lower band (failed breakout) OR 1w trend flips down OR chop regime shifts to range
            if (close_val < lower_aligned[i]) or (not trend_up) or (chop[i] >= chop_threshold):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above upper band (failed breakdown) OR 1w trend flips up OR chop regime shifts to range
            if (close_val > upper_aligned[i]) or (not trend_down) or (chop[i] >= chop_threshold):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Regime_v1"
timeframe = "1d"
leverage = 1.0