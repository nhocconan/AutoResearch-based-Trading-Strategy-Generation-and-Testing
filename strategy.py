#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_v2
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of the weekly Camarilla pivot bias.
Only long when price breaks above Donchian(20) high AND weekly bias is bullish (close > weekly H3 level).
Only short when price breaks below Donchian(20) low AND weekly bias is bearish (close < weekly L3 level).
Volume confirmation (volume > 1.5 * ATR6h) filters weak breakouts.
Uses discrete sizing 0.25 to limit fee drag. Target: 12-30 trades/year.
Weekly pivot bias provides structural edge in both bull and bear markets by aligning with higher-timeframe institutional levels.
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
    
    # Get weekly data for Camarilla pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla levels (H3/L3)
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla H3 = C + (H-L)*1.1/4
    # Camarilla L3 = C - (H-L)*1.1/4
    camarilla_h3_1w = c_1w + (h_1w - l_1w) * 1.1 / 4
    camarilla_l3_1w = c_1w - (h_1w - l_1w) * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR for volume confirmation (using 6h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and ATR(14)
    start_idx = max(lookback, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine weekly bias from Camarilla H3/L3
        # Bullish bias: weekly close > H3 (strong upside bias)
        # Bearish bias: weekly close < L3 (strong downside bias)
        weekly_close = c_1w[-1] if len(c_1w) > 0 else 0  # placeholder, will be replaced by aligned close
        # Use aligned weekly close for bias calculation
        df_1w_close = get_htf_data(prices, '1w')['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w_close)
        if np.isnan(weekly_close_aligned[i]):
            signals[i] = 0.0
            continue
            
        if weekly_close_aligned[i] > camarilla_h3_aligned[i]:
            weekly_bias = 'bullish'  # only allow longs
        elif weekly_close_aligned[i] < camarilla_l3_aligned[i]:
            weekly_bias = 'bearish'  # only allow shorts
        else:
            weekly_bias = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND volume confirm AND bullish weekly bias
            long_setup = (close[i] > highest[i]) and volume_confirm and (weekly_bias == 'bullish')
            
            # Short setup: price breaks below Donchian low AND volume confirm AND bearish weekly bias
            short_setup = (close[i] < lowest[i]) and volume_confirm and (weekly_bias == 'bearish')
            
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
            # Exit: price breaks below Donchian low OR weekly bias turns bearish
            if (close[i] < lowest[i]) or (weekly_bias == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR weekly bias turns bullish
            if (close[i] > highest[i]) or (weekly_bias == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_v2"
timeframe = "6h"
leverage = 1.0