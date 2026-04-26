#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_1dVolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakout aligned with 1-week EMA50 trend filter and 1-day volume spike (2.0x average) to capture medium-term swings in both bull and bear markets. Weekly trend ensures we trade with the dominant higher-timeframe momentum, while 6h Donchian breakouts provide precise entries. Volume confirmation reduces false breakouts. Discrete position sizing (0.25) and ATR-based stoploss (2.0x) manage risk. Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate volume MA(20) on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1w EMA(50), 1d volume MA, ATR, Donchian
    start_idx = max(50, 20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1w_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_1w_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        vol_spike = volume[i] > (2.0 * vol_ma_20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1w trend up AND volume spike
            long_signal = (close_val > highest_high[i]) and trend_1w_up and vol_spike
            
            # Short: price breaks below Donchian low AND 1w trend down AND volume spike
            short_signal = (close_val < lowest_low[i]) and trend_1w_down and vol_spike
            
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
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1w_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1w_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_1dVolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0