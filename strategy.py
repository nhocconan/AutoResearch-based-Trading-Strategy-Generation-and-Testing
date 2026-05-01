#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Uses 1d ADX(14) > 25 to identify trending markets (works in both bull/bear regimes).
# Long when price breaks above Donchian upper channel AND ADX > 25 AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower channel AND ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Volume confirmation and ADX filter reduce false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Donchian20_1dADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ and DM-
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Wilder's smoothing (first value is simple average)
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI+ and DI-
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    mask = atr != 0
    di_plus[mask] = (dm_plus_smooth[mask] / atr[mask]) * 100
    di_minus[mask] = (dm_minus_smooth[mask] / atr[mask]) * 100
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx_mask = (di_plus + di_minus) != 0
    dx[dx_mask] = (np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / 
                   (di_plus[dx_mask] + di_minus[dx_mask])) * 100
    
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    if len(dx) >= adx_period:
        # First ADX is simple average of first adx_period DX values
        valid_dx = dx[adx_period-1:2*adx_period-1]
        if not np.all(np.isnan(valid_dx)):
            adx[2*adx_period-2] = np.nanmean(valid_dx)
            
            for i in range(2*adx_period-1, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d trend: ADX > 25 indicates trending market
    adx_trending = adx_aligned > 25
    
    # Donchian Channel (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback*2, adx_period*2)  # warmup
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
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
        breakout_up = curr_high > highest_high[i]  # break above upper channel
        breakout_down = curr_low < lowest_low[i]   # break below lower channel
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper channel AND ADX > 25 AND volume confirmation
            if (breakout_up and 
                adx_trending[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower channel AND ADX > 25 AND volume confirmation
            elif (breakout_down and 
                  adx_trending[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower channel (stoploss) OR ADX < 20 (trend weakening)
            if (curr_low < lowest_low[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper channel (stoploss) OR ADX < 20 (trend weakening)
            if (curr_high > highest_high[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals