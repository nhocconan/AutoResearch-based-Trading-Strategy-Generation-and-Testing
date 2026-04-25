#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRRegime_TrendFilter
Hypothesis: On 4h timeframe, Donchian channel (20) breakouts capture strong momentum moves.
Entry requires: breakout above upper band (long) or below lower band (short) + ATR-based volatility regime filter (avoid choppy markets) + 1d EMA50 trend filter.
Exit: reverse Donchian breakout or trend filter failure.
Uses discrete position sizing (0.25) to limit trades (~20-50/year) and minimize fee drag.
Designed for BTC/ETH to work in both bull and bear markets by trading breakouts with trend and volatility regime confirmation, avoiding overtrading through tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Donchian channel (20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Donchian (20) + ATR (14) + EMA (50)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # ATR regime filter: avoid extreme volatility (chop) and too low volatility (dead market)
        # Use 1d ATR normalized by price: ATR/close
        atr_norm = atr_14_1d_aligned[i] / curr_close
        # Regime: 0.01 < ATR_norm < 0.05 (reasonable volatility for 4h/1d)
        vol_regime = (atr_norm > 0.01) and (atr_norm < 0.05)
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volatility regime + 1d uptrend
            long_breakout = (curr_high > highest_20[i]) and vol_regime and (curr_close > ema_50_1d_aligned[i])
            # Short: price breaks below Donchian lower band + volatility regime + 1d downtrend
            short_breakout = (curr_low < lowest_20[i]) and vol_regime and (curr_close < ema_50_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower band OR trend turns down
            if (curr_low < lowest_20[i]) or (curr_close < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper band OR trend turns up
            if (curr_high > highest_20[i]) or (curr_close > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRRegime_TrendFilter"
timeframe = "4h"
leverage = 1.0