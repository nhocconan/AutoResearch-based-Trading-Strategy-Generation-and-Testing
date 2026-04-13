#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
    # Long: Price breaks above Donchian(20) high AND 1d ADX > 25 AND volume > 1.5x 20-period average
    # Short: Price breaks below Donchian(20) low AND 1d ADX > 25 AND volume > 1.5x 20-period average
    # Exit: Price returns to Donchian(20) midpoint OR ADX < 20 (weak trend)
    # Using 4h for price action and volume, 1d only for ADX regime filter
    # Discrete position sizing (0.30) to balance return and drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d ADX to 4h (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            continue
        start_idx = i - lookback + 1
        highest_high[i] = np.max(high[start_idx:i+1])
        lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Calculate 4h volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20 - 1:
            continue
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    # Donchian midpoint for exit
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20  # exit condition
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i]  # price breaks above upper channel
        breakout_down = close[i] < lowest_low[i]  # price breaks below lower channel
        
        # Entry logic: Donchian breakout + strong trend + volume confirmation
        long_entry = breakout_up and strong_trend and volume_confirm
        short_entry = breakout_down and strong_trend and volume_confirm
        
        # Exit logic: price returns to midpoint OR weak trend
        long_exit = close[i] < donchian_mid[i] or weak_trend
        short_exit = close[i] > donchian_mid[i] or weak_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0