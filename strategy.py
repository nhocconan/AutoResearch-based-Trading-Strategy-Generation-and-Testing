#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20), 1d ADX > 25 (trending), and volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian lower (20), 1d ADX > 25, and volume > 1.5x 20-bar avg.
# Exit on opposite Donchian band touch or ADX < 20 (range regime).
# Donchian channels provide robust breakout levels; ADX filters for trending markets only.
# Volume confirmation reduces false breakouts. Timeframe: 4h.

name = "4h_Donchian20_1dADX_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        data = np.asarray(data)
        length = len(data)
        result = np.full(length, np.nan)
        if length < period:
            return result
        # first value: simple average
        result[period-1] = np.nanmean(data[:period])
        # rest: Wilder's smoothing
        for i in range(period, length):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20) on 4h
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, ADX > 25 (trending), volume spike
            if (curr_close > curr_donchian_upper and 
                curr_adx_1d > 25 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, ADX > 25 (trending), volume spike
            elif (curr_close < curr_donchian_lower and 
                  curr_adx_1d > 25 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price touches Donchian lower OR ADX < 20 (range)
            if (curr_low <= curr_donchian_lower or curr_adx_1d < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price touches Donchian upper OR ADX < 20 (range)
            if (curr_high >= curr_donchian_upper or curr_adx_1d < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals