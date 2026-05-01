#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1w ADX > 25 AND volume > 1.5x 20-bar average.
# Short when price breaks below 20-period Donchian low AND 1w ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture major trends with low trade frequency.
# Donchian channels provide clear breakout levels that work in both bull and bear markets.
# 1w ADX > 25 ensures we only trade strong trends, avoiding whipsaws in ranging markets.
# Volume confirmation reduces false breakouts and improves signal quality.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Donchian20_1wADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1w ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_gt_25 = adx > 25
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_gt_25)
    
    # Donchian(20) channels on 12h
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donch_high[i]  # break above upper channel
        breakout_down = curr_low < donch_low[i]  # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND 1w ADX > 25 AND volume confirmation
            if (breakout_up and 
                adx_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND 1w ADX > 25 AND volume confirmation
            elif (breakout_down and 
                  adx_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR 1w ADX falls below 20 (trend weakness)
            if (curr_low < donch_low[i] or 
                not adx_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR 1w ADX falls below 20 (trend weakness)
            if (curr_high > donch_high[i] or 
                not adx_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals