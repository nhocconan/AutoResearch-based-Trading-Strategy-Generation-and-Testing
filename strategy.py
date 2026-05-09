#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1d ADX trend filter
# Designed to capture strong trending moves while avoiding whipsaws in ranging markets
# Uses: Donchian(20) breakout for entries, 1d volume spike (>2x average) for confirmation,
# and 1d ADX(14) > 25 to ensure trending regime. Exits on opposite Donchian break.
# Target: 20-40 trades/year to minimize fee drag in 12h timeframe.

name = "12h_Donchian20_1dVolume_ADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    
    # Smoothed ADX
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_1d / vol_ma_20)
    
    # Calculate Donchian channels on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper and lower Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ratio = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = vol_ratio > 1.5
        # Trend filter: ADX > 25
        trend_filter = adx_val > 25
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume + trend
            if price > upper and volume_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + volume + trend
            elif price < lower and volume_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals