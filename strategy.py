#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Donchian(20) breakout + volume confirmation + 1d ADX trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average and 1d ADX > 25.
Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average and 1d ADX > 25.
Donchian channels capture volatility-based breakouts; volume confirmation reduces false signals; ADX filter ensures trending market.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get daily data for Donchian channels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).mean().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).mean().values
    # Donchian high = rolling max of high, Donchian low = rolling min of low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    # ADX requires +DI, -DI, and DX calculation
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DM_smooth = smoothed +DM, -DM_smooth = smoothed -DM, TR_smooth = smoothed TR
    # +DI = 100 * +DM_smooth / TR_smooth
    # -DI = 100 * -DM_smooth / TR_smooth
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    plus_dm = np.where((high_1d - prev_high) > (prev_low - low_1d), np.maximum(high_1d - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low_1d) > (high_1d - prev_high), np.maximum(prev_low - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    tr_smooth = np.zeros_like(tr)
    
    plus_dm_smooth[period-1] = np.nansum(plus_dm[:period])
    minus_dm_smooth[period-1] = np.nansum(minus_dm[:period])
    tr_smooth[period-1] = np.nansum(tr[:period])
    
    for i in range(period, len(tr)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[period-1] = np.nansum(dx[:period])
    for i in range(period, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align all to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_aligned[i]
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian low or trend weakens
            if (close[i] < donchian_low_aligned[i] or 
                adx_aligned[i] < 20):  # ADX < 20 indicates weak trend/no trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian high or trend weakens
            if (close[i] > donchian_high_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_ADX"
timeframe = "12h"
leverage = 1.0