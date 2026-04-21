#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian breakout with volume confirmation.
# In choppy markets (CHOP > 61.8), mean-revert at Donchian channels.
# In trending markets (CHOP < 38.2), breakout in direction of trend.
# Uses 1d ADX for trend confirmation and volume spike for confirmation.
# Target: 20-50 trades/year by combining regime filter with structure.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ADX on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([np.array([np.nan]), dm_plus])
    dm_minus = np.concatenate([np.array([np.nan]), dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(14, len(dx)):
        if i == 14:
            adx[i] = np.nanmean(dx[1:15])
        elif not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Choppiness Index
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # True Range for 4h
    tr1_4h = np.abs(high_4h[1:] - low_4h[1:])
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h = np.concatenate([np.array([np.nan]), tr_4h])
    
    # ATR for chop calculation
    atr_4h = np.full_like(tr_4h, np.nan)
    for i in range(1, len(tr_4h)):
        if i < 14:
            if i == 1:
                atr_4h[i] = tr_4h[i]
            else:
                valid = tr_4h[1:i+1][~np.isnan(tr_4h[1:i+1])]
                if len(valid) > 0:
                    atr_4h[i] = np.mean(valid)
        else:
            if not np.isnan(atr_4h[i-1]):
                atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Chop calculation
    sum_atr_14 = np.full_like(atr_4h, np.nan)
    for i in range(13, len(atr_4h)):
        if not np.isnan(atr_4h[i]):
            sum_atr_14[i] = np.nansum(atr_4h[i-13:i+1])
    
    # Max/min over 14 periods
    max_high_14 = np.full_like(high_4h, np.nan)
    min_low_14 = np.full_like(low_4h, np.nan)
    for i in range(13, len(high_4h)):
        max_high_14[i] = np.nanmax(high_4h[i-13:i+1])
        min_low_14[i] = np.nanmin(low_4h[i-13:i+1])
    
    chop = np.full_like(close_4h, np.nan)
    for i in range(13, len(close_4h)):
        if not np.isnan(sum_atr_14[i]) and not np.isnan(max_high_14[i]) and not np.isnan(min_low_14[i]):
            if max_high_14[i] != min_low_14[i]:
                chop[i] = 100 * np.log10(sum_atr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
    
    # Pre-calculate Donchian channels (20-period)
    donchian_high = np.full_like(close_4h, np.nan)
    donchian_low = np.full_like(close_4h, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.nanmax(high_4h[i-19:i+1])
        donchian_low[i] = np.nanmin(low_4h[i-19:i+1])
    
    # Volume average (20-period)
    vol_ma_20 = np.full_like(prices['volume'].values, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.nanmean(prices['volume'].values[i-19:i+1])
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20[i]
        
        # Regime filters
        adx_val = adx_1d_aligned[i]
        chop_val = chop[i]
        
        trending = adx_val > 25  # Trending market
        choppy = chop_val > 61.8  # Choppy market
        
        if position == 0:
            # Entry logic based on regime
            if choppy and volume_confirm:
                # Choppy market: mean reversion at Donchian boundaries
                if price <= donchian_low[i]:  # Near lower band -> long
                    signals[i] = 0.25
                    position = 1
                elif price >= donchian_high[i]:  # Near upper band -> short
                    signals[i] = -0.25
                    position = -1
            elif trending and volume_confirm:
                # Trending market: breakout in trend direction
                if price > donchian_high[i]:  # Break above -> long
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low[i]:  # Break below -> short
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian low OR chop increases significantly
                if price < donchian_low[i] or (choppy and chop_val > 70):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above Donchian high OR chop increases significantly
                if price > donchian_high[i] or (choppy and chop_val > 70):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Donchian_MeanRev_Breakout"
timeframe = "4h"
leverage = 1.0