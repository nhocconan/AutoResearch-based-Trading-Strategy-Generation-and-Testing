#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with daily volume confirmation and ADX trend filter
# Long when price breaks above 4h Donchian(20) high AND daily volume > 1.5x 20-period volume SMA AND ADX > 25
# Short when price breaks below 4h Donchian(20) low AND daily volume > 1.5x 20-period volume SMA AND ADX > 25
# Uses daily trend filter (ADX) to avoid counter-trend trades. Volume surge confirms breakout strength.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within optimal range.

name = "4h_donchian20_1d_vol_adx_v1"
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
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Daily volume SMA for confirmation
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_volume_series = pd.Series(daily_volume)
    daily_volume_sma = daily_volume_series.rolling(window=20, min_periods=20).mean().values
    daily_volume_sma_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_sma)
    
    # Daily ADX for trend filter (ADX > 25 = trending)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[atr_period-1] = np.mean(tr[:atr_period])
    dm_plus_smooth[atr_period-1] = np.mean(dm_plus[:atr_period])
    dm_minus_smooth[atr_period-1] = np.mean(dm_minus[:atr_period])
    
    for i in range(atr_period, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        dm_plus_smooth[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_smooth[i-1]
        dm_minus_smooth[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_smooth[i-1]
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[2*atr_period-1] = np.mean(dx[atr_period:2*atr_period])
    
    for i in range(2*atr_period, len(dx)):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align daily indicators to 4h timeframe
    daily_volume_sma_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_sma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if daily data not available
        if np.isnan(daily_volume_sma_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day SMA
        volume_confirmed = daily_volume[i] > 1.5 * daily_volume_sma_aligned[i]
        
        # Trend filter: ADX > 25
        trending = adx_aligned[i] > 25
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below 4h Donchian low OR volume dries up OR trend weakens
            if (close[i] <= donchian_low[i] or 
                not volume_confirmed or 
                not trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 4h Donchian high OR volume dries up OR trend weakens
            if (close[i] >= donchian_high[i] or 
                not volume_confirmed or 
                not trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 4h Donchian high AND volume confirmed AND trending
            if (close[i] > donchian_high[i] and 
                volume_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low AND volume confirmed AND trending
            elif (close[i] < donchian_low[i] and 
                  volume_confirmed and 
                  trending):
                signals[i] = -0.25
                position = -1
    
    return signals