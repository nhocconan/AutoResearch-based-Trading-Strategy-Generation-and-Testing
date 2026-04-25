#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRRegime_TrendFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with ATR-based regime filter and 1d EMA50 trend confirmation. In bullish 1d trend (close > EMA50), buy when price breaks above upper Donchian; in bearish 1d trend (close < EMA50), sell when price breaks below lower Donchian. ATR regime filter avoids whipsaw in choppy markets (ATR(14)/ATR(50) < 0.8 = chop, no trades). Volume confirmation (1.5x 20-bar avg) ensures participation. Discrete position sizing (0.25) targets ~20-40 trades/year. Designed to work in both bull and bear markets by following higher timeframe trend and avoiding chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: trending when ATR(14)/ATR(50) >= 0.8, choppy when < 0.8
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    trending_regime = atr_ratio >= 0.8
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for indicators
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend: price above/below EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation, trend alignment, and trending regime
            long_signal = (close[i] > highest_high[i] and 
                          volume_spike[i] and 
                          trend_bullish and 
                          trending_regime[i])
            short_signal = (close[i] < lowest_low[i] and 
                           volume_spike[i] and 
                           trend_bearish and 
                           trending_regime[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price breaks below lower Donchian or trend reverses or regime turns choppy
            exit_signal = (close[i] < lowest_low[i] or 
                          not trend_bullish or 
                          not trending_regime[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above upper Donchian or trend reverses or regime turns choppy
            exit_signal = (close[i] > highest_high[i] or 
                          not trend_bearish or 
                          not trending_regime[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRRegime_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0