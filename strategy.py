#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Uses 1d for trend direction (ADX > 25) and structure (Donchian channels).
# 12h timeframe for entry timing precision to avoid overtrading.
# Long when price breaks above Donchian upper channel with 1d ADX > 25 and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower channel with 1d ADX > 25 and volume confirmation.
# Discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

name = "12h_Donchian_20_1dADX_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX and Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilder_smooth(tr_1d, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[:14])  # First ADX value after 14 periods
    for i in range(14, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate ATR(14) for stoploss on 12h timeframe
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_12h = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels from 1d data (20-period)
    highest_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 30  # warmup for ADX and Donchian channels
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(adx_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_close > upper_channel_aligned[i]  # break above upper channel
        breakout_down = curr_close < lower_channel_aligned[i]  # break below lower channel
        
        # ADX trend filter: > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND strong trend AND volume confirmation
            if (breakout_up and 
                strong_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND strong trend AND volume confirmation
            elif (breakout_down and 
                  strong_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX falls below 20 (trend weakening)
            elif (curr_close < upper_channel_aligned[i] and curr_close > lower_channel_aligned[i]) or \
                 adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX falls below 20 (trend weakening)
            elif (curr_close < upper_channel_aligned[i] and curr_close > lower_channel_aligned[i]) or \
                 adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals