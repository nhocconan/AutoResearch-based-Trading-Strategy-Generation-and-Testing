#!/usr/bin/env python3
"""
4h_Vortex_Trend_With_RSI_Filter
Hypothesis: Combines Vortex Indicator (VI+) and (VI-) for trend direction with RSI(14) to avoid overbought/oversold extremes. Uses 1d ADX as regime filter to only trade when trending (ADX>25). Targets 20-30 trades/year via strict multi-condition entry. Works in bull (follow VI+) and bear (follow VI-) by aligning with trend direction. Vortex captures trend initiation, RSI prevents chasing extremes, ADX ensures trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def _wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # Initial value: simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0 (NaN)
        # Wilder's smoothing: previous * (period-1)/period + current/period
        for i in range(period, len(arr)):
            result[i] = result[i-1] * (period-1)/period + arr[i]/period
        return result
    
    tr_14 = _wilders_smoothing(tr, 14)
    dm_plus_14 = _wilders_smoothing(dm_plus, 14)
    dm_minus_14 = _wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _wilders_smoothing(dx, 14)
    adx_14 = adx
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Vortex Indicator on 4h data
    # VM+ = |current high - prior low|
    # VM- = |current low - prior high|
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # True Range for Vortex (same as above but for 4h)
    tr_vx1 = np.abs(high[1:] - low[1:])
    tr_vx2 = np.abs(high[1:] - close[:-1])
    tr_vx3 = np.abs(low[1:] - close[:-1])
    tr_vx = np.concatenate([[np.nan], np.maximum(tr_vx1, np.maximum(tr_vx2, tr_vx3))])
    
    # Sum over 14 periods
    def _sum_arr(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        for i in range(period-1, len(arr)):
            result[i] = np.nansum(arr[i-period+1:i+1])
        return result
    
    vm_plus_14 = _sum_arr(vm_plus, 14)
    vm_minus_14 = _sum_arr(vm_minus, 14)
    tr_vx_14 = _sum_arr(tr_vx, 14)
    
    # VI+ and VI-
    vi_plus = np.where(tr_vx_14 != 0, vm_plus_14 / tr_vx_14, 0)
    vi_minus = np.where(tr_vx_14 != 0, vm_minus_14 / tr_vx_14, 0)
    
    # Calculate RSI(14) on 4h close
    def _rsi(close_prices, period):
        delta = np.diff(close_prices)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(gain, np.nan, dtype=float)
        avg_loss = np.full_like(loss, np.nan, dtype=float)
        
        if len(gain) < period:
            return np.full_like(close_prices, np.nan, dtype=float)
            
        # Initial average
        avg_gain[period-1] = np.nansum(gain[1:period])  # Skip index 0 (NaN)
        avg_loss[period-1] = np.nansum(loss[1:period])
        
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = _rsi(close, 14)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = np.nan
    vol_ma_20[-10:] = np.nan
    # Recalculate properly with pandas for edge handling
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(rsi_14[i]) or np.isnan(adx_14_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        is_trending = adx_14_aligned[i] > 25
        
        # Vortex trend: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
        vx_uptrend = vi_plus[i] > vi_minus[i]
        vx_downtrend = vi_minus[i] > vi_plus[i]
        
        # RSI filter: avoid extremes (RSI < 30 oversold, > 70 overbought)
        rsi_not_extreme = (rsi_14[i] >= 30) and (rsi_14[i] <= 70)
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = vx_uptrend and rsi_not_extreme and vol_confirm and is_trending
        short_entry = vx_downtrend and rsi_not_extreme and vol_confirm and is_trending
        
        # Exit conditions: trend reversal or RSI extreme
        long_exit = (not vx_uptrend) or (rsi_14[i] > 70)
        short_exit = (not vx_downtrend) or (rsi_14[i] < 30)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0