#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_1dVolumeConfirmation_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly trend (price > weekly EMA50) and confirmed by daily volume spikes (>2.0x 20-day MA) provide edge in both bull and bear markets.
Weekly EMA50 filter ensures we only trade in the direction of the major trend, reducing whipsaws. Daily volume confirmation ensures breakouts have institutional participation.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets 12-30 trades/year (~50-120 over 4 years) to avoid fee drag on 6h timeframe.
ATR-based stoploss (2.0x) manages risk.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-day average volume on daily
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly EMA(50), daily vol MA(20), ATR(14), Donchian(20)
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
        vol_confirmed = volume[i] > (2.0 * vol_ma_20_1d_aligned[i])  # volume at least 2.0x daily average
        trend_up = close_val > ema_50_1w_aligned[i]
        trend_down = close_val < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly trend up AND volume confirmation
            long_signal = (close_val > highest_high[i]) and trend_up and vol_confirmed
            
            # Short: price breaks below Donchian lower AND weekly trend down AND volume confirmation
            short_signal = (close_val < lowest_low[i]) and trend_down and vol_confirmed
            
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
            if (not trend_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_1dVolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0