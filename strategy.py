#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# Long when price breaks above Donchian upper band + volume > 1.5x average + 1d ADX > 25
# Short when price breaks below Donchian lower band + volume > 1.5x average + 1d ADX > 25
# Exit when price crosses Donchian midline or volume drops
# Target: 75-200 total trades over 4 years (19-50/year) with strong trend filtering

name = "4h_donchian20_1d_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (14-period)
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initial values
    if len(tr) >= tr_period:
        atr[tr_period-1] = np.nanmean(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.nanmean(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.nanmean(dm_minus[1:tr_period])
        
        # Wilder's smoothing
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # Directional Indicators
    plus_di = np.full_like(atr, np.nan)
    minus_di = np.full_like(atr, np.nan)
    dx = np.full_like(atr, np.nan)
    
    mask = ~np.isnan(atr) & (atr != 0)
    plus_di[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    minus_di[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    
    dx_mask = ~np.isnan(plus_di) & ~np.isnan(minus_di) & ((plus_di + minus_di) != 0)
    dx[dx_mask] = 100 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    if len(dx) >= adx_period:
        valid_dx = dx[~np.isnan(dx)]
        if len(valid_dx) >= adx_period:
            adx[adx_period-1] = np.nanmean(valid_dx[:adx_period])
            for i in range(adx_period, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter
        trend_strong = adx_aligned[i] > 25
        
        if position == 1:  # long position
            # Exit: price crosses below midline OR volume drops OR trend weakens
            if (close[i] < donchian_middle[i] or not vol_confirm or not trend_strong):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above midline OR volume drops OR trend weakens
            if (close[i] > donchian_middle[i] or not vol_confirm or not trend_strong):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: break above upper band + volume + trend
            if (close[i] > donchian_upper[i] and vol_confirm and trend_strong):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + volume + trend
            elif (close[i] < donchian_lower[i] and vol_confirm and trend_strong):
                signals[i] = -0.25
                position = -1
    
    return signals