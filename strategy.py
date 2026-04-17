#!/usr/bin/env python3
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
    
    # === 1d ADX for trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing (14-period)
    period = 14
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    if len(tr) >= period + 1:
        # Initial ATR
        atr[period] = np.nanmean(tr[1:period+1])
        # Initial DM
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        for i in range(period + 1, len(tr)):
            # Wilder smoothing
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            
            # DI calculation
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
                # DX calculation
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        
        # ADX calculation (smoothed DX)
        adx_start = 2 * period
        if len(dx) > adx_start:
            adx[adx_start] = np.nanmean(dx[period+1:adx_start+1])
            for i in range(adx_start + 1, len(dx)):
                if not np.nan_to_num(dx[i], nan=0) == 0:
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # === 12h Donchian Channel (20-period) ===
    # Use 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_high = np.full_like(high_12h, np.nan)
    donch_low = np.full_like(low_12h, np.nan)
    
    lookback = 20
    for i in range(len(high_12h)):
        if i >= lookback - 1:
            donch_high[i] = np.max(high_12h[i-lookback+1:i+1])
            donch_low[i] = np.min(low_12h[i-lookback+1:i+1])
        elif i > 0:
            donch_high[i] = np.max(high_12h[0:i+1])
            donch_low[i] = np.min(low_12h[0:i+1])
        else:
            donch_high[i] = high_12h[0]
            donch_low[i] = low_12h[0]
    
    # === Align indicators to primary timeframe ===
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # === Volume confirmation (12h) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === ADX threshold for trending market ===
    ADX_THRESHOLD = 25
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] < ADX_THRESHOLD:
            # In ranging markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low
            elif close[i] < donch_low_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ADX_DonchianBreakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0