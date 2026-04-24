#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX regime filter (ADX > 25 = trending market).
- Entry: Long when price breaks above Donchian(20) high AND ADX > 25 AND volume > 1.5 * avg_volume(20).
         Short when price breaks below Donchian(20) low AND ADX > 25 AND volume > 1.5 * avg_volume(20).
- Exit: Opposite Donchian breakout OR ADX < 20 (range regime) OR volume < 0.5 * avg_volume(20).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels identify structural breakouts with clear support/resistance.
- ADX filters for trending markets to avoid whipsaws in ranging conditions.
- Volume confirmation ensures breakouts have participation.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 20-period average volume for confirmation
    avg_vol = np.full(n, np.nan)
    for i in range(20, n):
        avg_vol[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback-1)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            # Exit: Opposite breakout OR ADX < 20 (range) OR very low volume
            if position == 1:
                if (curr_close < lowest_low[i] or  # Opposite breakout
                    adx_1d_aligned[i] < 20 or      # Range regime
                    curr_volume < 0.5 * avg_vol[i]): # Low volume
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                if (curr_close > highest_high[i] or  # Opposite breakout
                    adx_1d_aligned[i] < 20 or      # Range regime
                    curr_volume < 0.5 * avg_vol[i]): # Low volume
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Long: Breakout above Donchian high + trending market (ADX>25) + volume confirmation
            if (curr_close > highest_high[i] and 
                adx_1d_aligned[i] > 25 and 
                curr_volume > 1.5 * avg_vol[i]):
                signals[i] = 0.30
                position = 1
            # Short: Breakout below Donchian low + trending market (ADX>25) + volume confirmation
            elif (curr_close < lowest_low[i] and 
                  adx_1d_aligned[i] > 25 and 
                  curr_volume > 1.5 * avg_vol[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1dADX_Regime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0