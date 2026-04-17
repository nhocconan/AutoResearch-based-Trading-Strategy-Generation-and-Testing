#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Uses price channels (Donchian) for entries, 1d ADX>25 for trend filter, volume spike for confirmation.
# Designed to work in bull (breakouts with trend) and bear (mean reversion via channel retracement).
# Target: 15-25 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ADX (14-period)
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
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    for i in range(len(tr)):
        if i < tr_period:
            tr_sum[i] = np.nan
            dm_plus_sum[i] = np.nan
            dm_minus_sum[i] = np.nan
        elif i == tr_period:
            tr_sum[i] = np.nansum(tr[i-tr_period+1:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-tr_period+1:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-tr_period+1:i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    # DI+, DI-, DX
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 
                  0)
    # ADX (smoothed DX)
    adx_period = 14
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < 2 * adx_period - 1:
            adx[i] = np.nan
        elif i == 2 * adx_period - 1:
            adx[i] = np.nanmean(dx[adx_period-1:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align daily Donchian and ADX to 4h
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need Donchian(20) and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_4h[i]) or 
            np.isnan(low_20_4h[i]) or 
            np.isnan(adx_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_4h[i] > 25
        
        # Price relative to daily Donchian channels
        price_above_high20 = close[i] > high_20_4h[i]
        price_below_low20 = close[i] < low_20_4h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and trending
            if (price_above_high20 and volume_filter and trending):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and trending
            elif (price_below_low20 and volume_filter and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price retracement to midline OR ADX weakens
            midline = (high_20_4h[i] + low_20_4h[i]) / 2
            if (close[i] < midline) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price retracement to midline OR ADX weakens
            midline = (high_20_4h[i] + low_20_4h[i]) / 2
            if (close[i] > midline) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX25_Volume"
timeframe = "4h"
leverage = 1.0